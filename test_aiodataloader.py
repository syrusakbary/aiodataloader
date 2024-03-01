from asyncio import gather
from functools import partial
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple, TypeVar, Union

import pytest
import asyncio

from aiodataloader import DataLoader

pytestmark = pytest.mark.asyncio

T1 = TypeVar("T1")
T2 = TypeVar("T2")


async def do_test() -> bool:
    return True


def id_loader(
    *,
    resolve: Optional[
        Callable[[List[T1]], Coroutine[Any, Any, Union[List[T1], List[T2]]]]
    ] = None,
    **dl_kwargs: Any,
) -> Tuple[DataLoader[T1, Union[T1, T2]], List[List[T1]]]:
    load_calls: List[List[T1]] = []

    async def default_resolve(x: List[T1]) -> List[T1]:
        return x

    async def fn(keys: List[T1]) -> Union[List[T1], List[T2]]:
        load_calls.append(keys)
        return await (resolve or default_resolve)(keys)

    identity_loader: DataLoader[Any, Any] = DataLoader(fn, **dl_kwargs)
    return identity_loader, load_calls


async def test_build_a_simple_data_loader() -> None:
    async def call_fn(keys: List[int]) -> List[int]:
        return keys

    identity_loader = DataLoader(call_fn)

    promise1 = identity_loader.load(1)

    value1 = await promise1
    assert value1 == 1


async def test_can_build_a_data_loader_from_a_partial() -> None:
    value_map = {1: "one"}

    async def call_fn(context: Dict[int, T1], keys: List[int]) -> List[Optional[T1]]:
        return [context.get(key) for key in keys]

    partial_fn = partial(call_fn, value_map)
    identity_loader = DataLoader(partial_fn)

    promise1 = identity_loader.load(1)

    value1 = await promise1
    assert value1 == "one"


async def test_supports_loading_multiple_keys_in_one_call() -> None:
    async def call_fn(keys: List[int]) -> List[int]:
        return keys

    identity_loader = DataLoader(call_fn)

    promise_all = identity_loader.load_many([1, 2])

    values = await promise_all
    assert values == [1, 2]

    promise_all = identity_loader.load_many([])

    values = await promise_all
    assert values == []


async def test_batches_multiple_requests() -> None:
    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader()
    identity_loader, load_calls = loader_result

    promise1 = identity_loader.load(1)
    promise2 = identity_loader.load(2)

    p = gather(promise1, promise2)

    value1, value2 = await p

    assert value1 == 1
    assert value2 == 2

    assert load_calls == [[1, 2]]


async def test_batches_multiple_requests_with_max_batch_sizes() -> None:
    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader(
        max_batch_size=2
    )
    identity_loader, load_calls = loader_result

    promise1 = identity_loader.load(1)
    promise2 = identity_loader.load(2)
    promise3 = identity_loader.load(3)

    p = gather(promise1, promise2, promise3)

    value1, value2, value3 = await p

    assert value1 == 1
    assert value2 == 2
    assert value3 == 3

    assert load_calls == [[1, 2], [3]]


async def test_coalesces_identical_requests() -> None:
    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader()
    identity_loader, load_calls = loader_result

    promise1 = identity_loader.load(1)
    promise2 = identity_loader.load(1)

    assert promise1 == promise2
    p = gather(promise1, promise2)

    value1, value2 = await p

    assert value1 == 1
    assert value2 == 1

    assert load_calls == [[1]]


async def test_caches_repeated_requests() -> None:
    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader()
    identity_loader, load_calls = loader_result

    a, b = await gather(identity_loader.load("A"), identity_loader.load("B"))

    assert a == "A"
    assert b == "B"

    assert load_calls == [["A", "B"]]

    a2, c = await gather(identity_loader.load("A"), identity_loader.load("C"))

    assert a2 == "A"
    assert c == "C"

    assert load_calls == [["A", "B"], ["C"]]

    a3, b2, c2 = await gather(
        identity_loader.load("A"), identity_loader.load("B"), identity_loader.load("C")
    )

    assert a3 == "A"
    assert b2 == "B"
    assert c2 == "C"

    assert load_calls == [["A", "B"], ["C"]]


async def test_clears_single_value_in_loader() -> None:
    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader()
    identity_loader, load_calls = loader_result

    a, b = await gather(identity_loader.load("A"), identity_loader.load("B"))

    assert a == "A"
    assert b == "B"

    assert load_calls == [["A", "B"]]

    identity_loader.clear("A")

    a2, b2 = await gather(identity_loader.load("A"), identity_loader.load("B"))

    assert a2 == "A"
    assert b2 == "B"

    assert load_calls == [["A", "B"], ["A"]]


async def test_clears_all_values_in_loader() -> None:
    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader()
    identity_loader, load_calls = loader_result

    a, b = await gather(identity_loader.load("A"), identity_loader.load("B"))

    assert a == "A"
    assert b == "B"

    assert load_calls == [["A", "B"]]

    identity_loader.clear_all()

    a2, b2 = await gather(identity_loader.load("A"), identity_loader.load("B"))

    assert a2 == "A"
    assert b2 == "B"

    assert load_calls == [["A", "B"], ["A", "B"]]


async def test_allows_priming_the_cache() -> None:
    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader()
    identity_loader, load_calls = loader_result

    identity_loader.prime("A", "A")

    a, b = await gather(identity_loader.load("A"), identity_loader.load("B"))

    assert a == "A"
    assert b == "B"

    assert load_calls == [["B"]]


async def test_does_not_prime_keys_that_already_exist() -> None:
    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader()
    identity_loader, load_calls = loader_result

    identity_loader.prime("A", "X")

    a1 = await identity_loader.load("A")
    b1 = await identity_loader.load("B")

    assert a1 == "X"
    assert b1 == "B"

    identity_loader.prime("A", "Y")
    identity_loader.prime("B", "Y")

    a2 = await identity_loader.load("A")
    b2 = await identity_loader.load("B")

    assert a2 == "X"
    assert b2 == "B"

    assert load_calls == [["B"]]


# # Represents Errors


async def test_resolves_to_error_to_indicate_failure() -> None:
    async def resolve(keys: List[int]) -> List[int]:
        mapped_keys = [
            key if key % 2 == 0 else Exception("Odd: {}".format(key)) for key in keys
        ]
        # ignored because Exceptions are not expected for a batch_load_fn
        # but we are testing unexpected behavior
        return mapped_keys  # type: ignore

    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader(
        resolve=resolve
    )
    even_loader, load_calls = loader_result

    with pytest.raises(Exception) as exc_info:
        await even_loader.load(1)

    assert str(exc_info.value) == "Odd: 1"

    value2 = await even_loader.load(2)
    assert value2 == 2
    assert load_calls == [[1], [2]]


async def test_can_represent_failures_and_successes_simultaneously() -> None:
    async def resolve(keys: List[int]) -> List[int]:
        mapped_keys = [
            key if key % 2 == 0 else Exception("Odd: {}".format(key)) for key in keys
        ]
        return mapped_keys  # type: ignore

    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader(
        resolve=resolve
    )
    even_loader, load_calls = loader_result

    promise1 = even_loader.load(1)
    promise2 = even_loader.load(2)

    with pytest.raises(Exception) as exc_info:
        await promise1

    assert str(exc_info.value) == "Odd: 1"
    value2 = await promise2
    assert value2 == 2
    assert load_calls == [[1, 2]]


async def test_caches_failed_fetches() -> None:
    async def resolve(keys: List[int]) -> List[int]:
        mapped_keys = [Exception("Error: {}".format(key)) for key in keys]
        return mapped_keys  # type: ignore

    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader(
        resolve=resolve
    )
    error_loader, load_calls = loader_result

    with pytest.raises(Exception) as exc_info:
        await error_loader.load(1)

    assert str(exc_info.value) == "Error: 1"

    with pytest.raises(Exception) as exc_info:
        await error_loader.load(1)

    assert str(exc_info.value) == "Error: 1"

    assert load_calls == [[1]]


async def test_caches_failed_fetches_2() -> None:
    loader_result: Tuple[DataLoader[int, int], List[List[int]]] = id_loader()
    identity_loader, load_calls = loader_result

    identity_loader.prime(1, Exception("Error: 1"))  # type: ignore

    with pytest.raises(Exception):
        await identity_loader.load(1)

    assert load_calls == []


# It is resilient to job queue ordering
async def test_batches_loads_occuring_within_promises() -> None:
    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader()
    identity_loader, load_calls = loader_result

    async def load_b_1() -> str:
        return await load_b_2()

    async def load_b_2() -> str:
        return await identity_loader.load("B")

    values = list(await gather(identity_loader.load("A"), load_b_1()))

    assert values == ["A", "B"]

    assert load_calls == [["A", "B"]]


async def test_catches_error_if_loader_resolver_fails() -> None:
    exc = Exception("AOH!")

    def do_resolve(x: List[Any]) -> Coroutine[Any, Any, List[Any]]:
        raise exc

    loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader(
        resolve=do_resolve
    )
    a_loader, a_load_calls = loader_result

    with pytest.raises(Exception) as exc_info:
        await a_loader.load("A1")

    assert exc_info.value == exc


async def test_can_call_a_loader_from_a_loader() -> None:
    deep_loader_result: Tuple[
        DataLoader[Tuple[str, ...], Tuple[str, ...]], List[List[Tuple[str, ...]]]
    ] = id_loader()
    deep_loader, deep_load_calls = deep_loader_result

    async def do_resolve(keys: List[str]) -> List[str]:
        return list(await deep_loader.load(tuple(keys)))

    a_loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader(
        resolve=do_resolve
    )
    a_loader, a_load_calls = a_loader_result

    b_loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader(
        resolve=do_resolve
    )
    b_loader, b_load_calls = b_loader_result

    a1, b1, a2, b2 = await gather(
        a_loader.load("A1"),
        b_loader.load("B1"),
        a_loader.load("A2"),
        b_loader.load("B2"),
    )

    assert a1 == "A1"
    assert b1 == "B1"
    assert a2 == "A2"
    assert b2 == "B2"

    assert a_load_calls == [["A1", "A2"]]
    assert b_load_calls == [["B1", "B2"]]
    assert deep_load_calls == [[("A1", "A2"), ("B1", "B2")]]


async def test_dataloader_clear_with_missing_key_works() -> None:
    async def do_resolve(x: List[Any]) -> List[Any]:
        return x

    a_loader_result: Tuple[DataLoader[str, str], List[List[str]]] = id_loader(
        resolve=do_resolve
    )
    a_loader, a_load_calls = a_loader_result

    assert a_loader.clear("A1") == a_loader


async def test_load_no_key() -> None:
    async def call_fn(keys: List[int]) -> List[int]:
        return keys

    identity_loader = DataLoader(call_fn)
    with pytest.raises(TypeError):
        identity_loader.load()  # type: ignore


async def test_load_none() -> None:
    async def call_fn(keys: List[int]) -> List[int]:
        return keys

    identity_loader = DataLoader(call_fn)
    with pytest.raises(TypeError):
        identity_loader.load(None)  # type: ignore

async def main():
    await test_build_a_simple_data_loader()
    await test_can_build_a_data_loader_from_a_partial()
    await test_supports_loading_multiple_keys_in_one_call()
    await test_batches_multiple_requests()
    await test_batches_multiple_requests_with_max_batch_sizes()
    await test_coalesces_identical_requests()
    await test_caches_repeated_requests()
    await test_clears_single_value_in_loader()
    await test_clears_all_values_in_loader()
    await test_allows_priming_the_cache()
    await test_does_not_prime_keys_that_already_exist()
    await test_resolves_to_error_to_indicate_failure()
    await test_can_represent_failures_and_successes_simultaneously()
    await test_caches_failed_fetches()
    await test_caches_failed_fetches_2()
    await test_batches_loads_occuring_within_promises()
    await test_catches_error_if_loader_resolver_fails()
    await test_can_call_a_loader_from_a_loader()
    await test_dataloader_clear_with_missing_key_works()
    await test_load_no_key()
    await test_load_none()

asyncio.run(main())

