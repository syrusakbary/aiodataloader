from asyncio import (
    AbstractEventLoop,
    Future,
    gather,
    ensure_future,
    get_event_loop,
    iscoroutine,
    iscoroutinefunction,
)
from functools import partial

from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)
from typing_extensions import Protocol, TypeGuard

__version__ = "0.2.1"


def iscoroutinefunctionorpartial(fn: Callable) -> TypeGuard[Callable[..., Coroutine]]:
    return iscoroutinefunction(fn.func if isinstance(fn, partial) else fn)


KeyT = TypeVar("KeyT")
ReturnT = TypeVar("ReturnT")
CacheKeyT = TypeVar("CacheKeyT")
T = TypeVar("T")


class BatchLoadFnProto(Protocol[KeyT, ReturnT]):
    async def __call__(self, keys: List[KeyT]) -> List[ReturnT]:
        ...


class Loader(tuple, Generic[KeyT, ReturnT]):
    # NamedTuple has a metaclass conflict with Generic that prevents
    # it from being used generically, and dataclasses aren't supported until
    # Python 3.7

    key: KeyT
    future: "Future[ReturnT]"

    def __new__(cls, key: KeyT, future: "Future[ReturnT]"):
        self = tuple.__new__(cls, (key, future))
        self.key = key
        self.future = future
        return self

    def __repr__(self) -> str:
        return f"Loader(key={self.key}, future={self.future})"


class DataLoader(Generic[KeyT, ReturnT]):

    batch: bool = True
    max_batch_size: Optional[int] = None
    cache: Optional[bool] = True
    _queue: List[Loader[KeyT, ReturnT]]

    def __init__(
        self,
        batch_load_fn: Optional[BatchLoadFnProto[KeyT, ReturnT]] = None,
        batch: Optional[bool] = None,
        max_batch_size: Optional[int] = None,
        cache: Optional[bool] = None,
        get_cache_key: Optional[Callable[[KeyT], CacheKeyT]] = None,
        cache_map: Optional[Dict[Union[CacheKeyT, KeyT], Any]] = None,
        loop: Optional[AbstractEventLoop] = None,
    ):
        self.loop = loop or get_event_loop()

        if batch_load_fn is not None:
            self.batch_load_fn = batch_load_fn

        assert iscoroutinefunctionorpartial(self.batch_load_fn), "batch_load_fn must be coroutine. Received: {}".format(
            self.batch_load_fn)

        if not callable(self.batch_load_fn):
            raise TypeError((
                'DataLoader must be have a batch_load_fn which accepts '
                'Iterable<key> and returns Future<Iterable<value>>, but got: {}.'
            ).format(batch_load_fn))

        if batch is not None:
            self.batch = batch

        if max_batch_size is not None:
            self.max_batch_size = max_batch_size

        if cache is not None:
            self.cache = cache

        self.get_cache_key = get_cache_key or (lambda x: x)

        self._cache = cache_map if cache_map is not None else {}
        self._queue = []

    def load(self, key: Optional[KeyT] = None) -> "Future[ReturnT]":
        """
        Loads a key, returning a `Future` for the value represented by that key.
        """
        if key is None:
            raise TypeError((
                'The loader.load() function must be called with a value, '
                'but got: {}.'
            ).format(key))

        cache_key = self.get_cache_key(key)

        # If caching and there is a cache-hit, return cached Future.
        if self.cache:
            cached_result = self._cache.get(cache_key)
            if cached_result:
                return cached_result

        # Otherwise, produce a new Future for this value.
        future = self.loop.create_future()
        # If caching, cache this Future.
        if self.cache:
            self._cache[cache_key] = future

        self.do_resolve_reject(key, future)
        return future

    def do_resolve_reject(self, key: KeyT, future: "Future[ReturnT]") -> None:
        # Enqueue this Future to be dispatched.
        self._queue.append(Loader(
            key=key,
            future=future
        ))
        # Determine if a dispatch of this queue should be scheduled.
        # A single dispatch should be scheduled per queue at the time when the
        # queue changes from "empty" to "full".
        if len(self._queue) == 1:
            if self.batch:
                # If batching, schedule a task to dispatch the queue.
                enqueue_post_future_job(self.loop, self)
            else:
                # Otherwise dispatch the (queue of one) immediately.
                dispatch_queue(self)

    def load_many(self, keys: Iterable[KeyT]) -> "Future[List[ReturnT]]":
        """
        Loads multiple keys, returning a list of values

        >>> a, b = await my_loader.load_many([ 'a', 'b' ])

        This is equivalent to the more verbose:

        >>> a, b = await gather(
        >>>    my_loader.load('a'),
        >>>    my_loader.load('b')
        >>> )
        """
        if not isinstance(keys, Iterable):
            raise TypeError((
                'The loader.load_many() function must be called with Iterable<key> '
                'but got: {}.'
            ).format(keys))

        return gather(*[self.load(key) for key in keys])

    def clear(self, key: KeyT) -> "DataLoader":
        """
        Clears the value at `key` from the cache, if it exists. Returns itself for
        method chaining.
        """
        cache_key = self.get_cache_key(key)
        self._cache.pop(cache_key, None)
        return self

    def clear_all(self) -> "DataLoader":
        """
        Clears the entire cache. To be used when some event results in unknown
        invalidations across this particular `DataLoader`. Returns itself for
        method chaining.
        """
        self._cache.clear()
        return self

    def prime(self, key: KeyT, value: ReturnT) -> "DataLoader":
        """
        Adds the provied key and value to the cache. If the key already exists, no
        change is made. Returns itself for method chaining.
        """
        cache_key = self.get_cache_key(key)

        # Only add the key if it does not already exist.
        if cache_key not in self._cache:
            # Cache a rejected future if the value is an Error, in order to match
            # the behavior of load(key).
            future = self.loop.create_future()
            if isinstance(value, Exception):
                future.set_exception(value)
            else:
                future.set_result(value)

            self._cache[cache_key] = future

        return self


def enqueue_post_future_job(loop: AbstractEventLoop, loader: DataLoader) -> None:
    async def dispatch() -> None:
        dispatch_queue(loader)

    loop.call_soon(ensure_future, dispatch())


def get_chunks(iterable_obj: List[T], chunk_size: int = 1) -> Iterator[List[T]]:
    chunk_size = max(1, chunk_size)
    return (iterable_obj[i:i + chunk_size] for i in range(0, len(iterable_obj), chunk_size))


def dispatch_queue(loader: DataLoader) -> None:
    """
    Given the current state of a Loader instance, perform a batch load
    from its current queue.
    """
    # Take the current loader queue, replacing it with an empty queue.
    queue = loader._queue
    loader._queue = []

    # If a max_batch_size was provided and the queue is longer, then segment the
    # queue into multiple batches, otherwise treat the queue as a single batch.
    max_batch_size = loader.max_batch_size

    if max_batch_size and max_batch_size < len(queue):
        chunks = get_chunks(queue, max_batch_size)
        for chunk in chunks:
            ensure_future(dispatch_queue_batch(
                loader,
                chunk
            ))
    else:
        ensure_future(dispatch_queue_batch(loader, queue))


async def dispatch_queue_batch(
    loader: DataLoader[KeyT, ReturnT], queue: List[Loader[KeyT, ReturnT]]
) -> None:
    # Collect all keys to be loaded in this dispatch
    keys = [ql.key for ql in queue]

    # Call the provided batch_load_fn for this loader with the loader queue's keys.
    batch_future = loader.batch_load_fn(keys)

    # Assert the expected response from batch_load_fn
    if not batch_future or not iscoroutine(batch_future):
        return failed_dispatch(
            loader,
            queue,
            TypeError((
                'DataLoader must be constructed with a function which accepts '
                'Iterable<key> and returns Future<Iterable<value>>, but the function did '
                'not return a Coroutine: {}.'
            ).format(batch_future))
        )

    try:
        values = await batch_future
        if not isinstance(values, Iterable):
            raise TypeError((
                'DataLoader must be constructed with a function which accepts '
                'Iterable<key> and returns Future<Iterable<value>>, but the function did '
                'not return a Future of a Iterable: {}.'
            ).format(values))

        values = list(values)
        if len(values) != len(keys):
            raise TypeError((
                'DataLoader must be constructed with a function which accepts '
                'Iterable<key> and returns Future<Iterable<value>>, but the function did '
                'not return a Future of a Iterable with the same length as the Iterable '
                'of keys.'
                '\n\nKeys:\n{}'
                '\n\nValues:\n{}'
            ).format(keys, values))

        # Step through the values, resolving or rejecting each Future in the
        # loaded queue.
        for ql, value in zip(queue, values):
            if isinstance(value, Exception):
                ql.future.set_exception(value)
            else:
                ql.future.set_result(value)

    except Exception as e:
        return failed_dispatch(loader, queue, e)


def failed_dispatch(
    loader: DataLoader[KeyT, ReturnT],
    queue: List[Loader[KeyT, ReturnT]],
    error: Exception,
) -> None:
    """
    Do not cache individual loads if the entire batch dispatch fails,
    but still reject each request so they do not hang.
    """
    for ql in queue:
        loader.clear(ql.key)
        ql.future.set_exception(error)
