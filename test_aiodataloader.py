from collections.abc import Callable, Coroutine
import pytest
from asyncio import gather
from functools import partial
from pytest import raises
from typing import Dict, List, Optional, Tuple, TypeVar
from aiodataloader import DataLoader

pytestmark = pytest.mark.asyncio


T1 = TypeVar("T1")
T2 = TypeVar("T2")


async def do_test():
    return True


def id_loader(
    *, resolve: Optional[Callable[..., Coroutine]] = None, **dl_kwargs
) -> Tuple[DataLoader, List]:
    load_calls = []

    async def default_resolve(x: T1) -> T1:
        return x

    if resolve is None:
        resolve = default_resolve

    async def fn(keys: List) -> List:
        load_calls.append(keys)
        return await resolve(keys)

    identity_loader: DataLoader = DataLoader(fn, **dl_kwargs)
    return identity_loader, load_calls


async def test_build_a_simple_data_loader():
    async def call_fn(keys: List[int]) -> List[int]:
        return keys

    identity_loader = DataLoader(call_fn)

    promise1 = identity_loader.load(1)

    value1 = await promise1
    assert value1 == 1


async def test_can_build_a_data_loader_from_a_partial():
    value_map = {1: "one"}

    async def call_fn(context: Dict, keys: List[int]):
        return [context.get(key) for key in keys]

    partial_fn = partial(call_fn, value_map)
    identity_loader = DataLoader(partial_fn)

    promise1 = identity_loader.load(1)

    value1 = await promise1
    assert value1 == 'one'


async def test_supports_loading_multiple_keys_in_one_call():
    async def call_fn(keys: List[int]):
        return keys

    identity_loader = DataLoader(call_fn)

    promise_all = identity_loader.load_many([1, 2])

    values = await promise_all
    assert values == [1, 2]

    promise_all = identity_loader.load_many([])

    values = await promise_all
    assert values == []


async def test_batches_multiple_requests():
    identity_loader, load_calls = id_loader()

    promise1 = identity_loader.load(1)
    promise2 = identity_loader.load(2)

    p = gather(promise1, promise2)

    value1, value2 = await p

    assert value1 == 1
    assert value2 == 2

    assert load_calls == [[1, 2]]


async def test_batches_multiple_requests_with_max_batch_sizes():
    identity_loader, load_calls = id_loader(max_batch_size=2)

    promise1 = identity_loader.load(1)
    promise2 = identity_loader.load(2)
    promise3 = identity_loader.load(3)

    p = gather(promise1, promise2, promise3)

    value1, value2, value3 = await p

    assert value1 == 1
    assert value2 == 2
    assert value3 == 3

    assert load_calls == [[1, 2], [3]]


async def test_coalesces_identical_requests():
    identity_loader, load_calls = id_loader()

    promise1 = identity_loader.load(1)
    promise2 = identity_loader.load(1)

    assert promise1 == promise2
    p = gather(promise1, promise2)

    value1, value2 = await p

    assert value1 == 1
    assert value2 == 1

    assert load_calls == [[1]]


async def test_caches_repeated_requests():
    identity_loader, load_calls = id_loader()

    a, b = await gather(
        identity_loader.load('A'),
        identity_loader.load('B')
    )

    assert a == 'A'
    assert b == 'B'

    assert load_calls == [['A', 'B']]

    a2, c = await gather(
        identity_loader.load('A'),
        identity_loader.load('C')
    )

    assert a2 == 'A'
    assert c == 'C'

    assert load_calls == [['A', 'B'], ['C']]

    a3, b2, c2 = await gather(
        identity_loader.load('A'),
        identity_loader.load('B'),
        identity_loader.load('C')
    )

    assert a3 == 'A'
    assert b2 == 'B'
    assert c2 == 'C'

    assert load_calls == [['A', 'B'], ['C']]


async def test_clears_single_value_in_loader():
    identity_loader, load_calls = id_loader()

    a, b = await gather(
        identity_loader.load('A'),
        identity_loader.load('B')
    )

    assert a == 'A'
    assert b == 'B'

    assert load_calls == [['A', 'B']]

    identity_loader.clear('A')

    a2, b2 = await gather(
        identity_loader.load('A'),
        identity_loader.load('B')
    )

    assert a2 == 'A'
    assert b2 == 'B'

    assert load_calls == [['A', 'B'], ['A']]


async def test_clears_all_values_in_loader():
    identity_loader, load_calls = id_loader()

    a, b = await gather(
        identity_loader.load('A'),
        identity_loader.load('B')
    )

    assert a == 'A'
    assert b == 'B'

    assert load_calls == [['A', 'B']]

    identity_loader.clear_all()

    a2, b2 = await gather(
        identity_loader.load('A'),
        identity_loader.load('B')
    )

    assert a2 == 'A'
    assert b2 == 'B'

    assert load_calls == [['A', 'B'], ['A', 'B']]


async def test_allows_priming_the_cache():
    identity_loader, load_calls = id_loader()

    identity_loader.prime('A', 'A')

    a, b = await gather(
        identity_loader.load('A'),
        identity_loader.load('B')
    )

    assert a == 'A'
    assert b == 'B'

    assert load_calls == [['B']]


async def test_does_not_prime_keys_that_already_exist():
    identity_loader, load_calls = id_loader()

    identity_loader.prime('A', 'X')

    a1 = await identity_loader.load('A')
    b1 = await identity_loader.load('B')

    assert a1 == 'X'
    assert b1 == 'B'

    identity_loader.prime('A', 'Y')
    identity_loader.prime('B', 'Y')

    a2 = await identity_loader.load('A')
    b2 = await identity_loader.load('B')

    assert a2 == 'X'
    assert b2 == 'B'

    assert load_calls == [['B']]


# # Represents Errors

async def test_resolves_to_error_to_indicate_failure():
    async def resolve(keys):
        mapped_keys = [
            key if key % 2 == 0 else Exception("Odd: {}".format(key))
            for key in keys
        ]
        return mapped_keys

    even_loader, load_calls = id_loader(resolve=resolve)

    with raises(Exception) as exc_info:
        await even_loader.load(1)

    assert str(exc_info.value) == "Odd: 1"

    value2 = await even_loader.load(2)
    assert value2 == 2
    assert load_calls == [[1], [2]]


async def test_can_represent_failures_and_successes_simultaneously():
    async def resolve(keys):
        mapped_keys = [
            key if key % 2 == 0 else Exception("Odd: {}".format(key))
            for key in keys
        ]
        return mapped_keys
    even_loader, load_calls = id_loader(resolve=resolve)

    promise1 = even_loader.load(1)
    promise2 = even_loader.load(2)

    with raises(Exception) as exc_info:
        await promise1

    assert str(exc_info.value) == "Odd: 1"
    value2 = await promise2
    assert value2 == 2
    assert load_calls == [[1, 2]]


async def test_caches_failed_fetches():
    async def resolve(keys):
        mapped_keys = [
            Exception("Error: {}".format(key))
            for key in keys
        ]
        return mapped_keys
    error_loader, load_calls = id_loader(resolve=resolve)

    with raises(Exception) as exc_info:
        await error_loader.load(1)

    assert str(exc_info.value) == "Error: 1"

    with raises(Exception) as exc_info:
        await error_loader.load(1)

    assert str(exc_info.value) == "Error: 1"

    assert load_calls == [[1]]


async def test_caches_failed_fetches_2():
    identity_loader, load_calls = id_loader()

    identity_loader.prime(1, Exception("Error: 1"))

    with raises(Exception) as exc_info:
        await identity_loader.load(1)

    assert load_calls == []

# It is resilient to job queue ordering

async def test_batches_loads_occuring_within_promises():
    identity_loader, load_calls = id_loader()
    async def load_b_1():
        return await load_b_2()

    async def load_b_2():
        return await identity_loader.load('B')

    values = await gather(
        identity_loader.load('A'),
        load_b_1()
    )

    assert values == ['A', 'B']

    assert load_calls == [['A', 'B']]


async def test_catches_error_if_loader_resolver_fails():
    exc = Exception("AOH!")
    def do_resolve(x):
        raise exc

    a_loader, a_load_calls = id_loader(resolve=do_resolve)

    with raises(Exception) as exc_info:
        await a_loader.load('A1')

    assert exc_info.value == exc


async def test_can_call_a_loader_from_a_loader():
    deep_loader, deep_load_calls = id_loader()

    async def do_resolve(keys):
        return await deep_loader.load(tuple(keys))

    a_loader, a_load_calls = id_loader(resolve=do_resolve)
    b_loader, b_load_calls = id_loader(resolve=do_resolve)

    a1, b1, a2, b2 = await gather(
        a_loader.load('A1'),
        b_loader.load('B1'),
        a_loader.load('A2'),
        b_loader.load('B2')
    )

    assert a1 == 'A1'
    assert b1 == 'B1'
    assert a2 == 'A2'
    assert b2 == 'B2'

    assert a_load_calls == [['A1', 'A2']]
    assert b_load_calls == [['B1', 'B2']]
    assert deep_load_calls == [[('A1', 'A2'), ('B1', 'B2')]]


async def test_dataloader_clear_with_missing_key_works():
    async def do_resolve(x):
        return x

    a_loader, a_load_calls = id_loader(resolve=do_resolve)
    assert a_loader.clear('A1') == a_loader
