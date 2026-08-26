# From Zero to Interview-Ready: Async, Threading, Multiprocessing & FastAPI

A no-gaps roadmap. Each module has: **what to learn**, **why it matters**, a **hands-on task**, and **checkpoint questions** you should be able to answer before moving on. Don't skip the checkpoints — they're the interview questions in disguise.

---

## Module 0: Foundations (before touching concurrency at all)

You cannot understand *why* async/threading/multiprocessing exist without understanding what a program is actually doing while it runs.

**Learn:**
- What a process is (memory space, PID, resources)
- What a thread is (shares memory with its process)
- CPU-bound vs I/O-bound work — this distinction drives every decision later
- What "blocking" means at the OS level (a syscall like `read()` that waits)
- What the CPU scheduler does, and what "context switching" means
- What the GIL (Global Interpreter Lock) is, in one sentence, and *why* CPython has it

**Hands-on:** Write a script that's clearly CPU-bound (e.g. sum primes up to 10 million) and one that's clearly I/O-bound (e.g. hit a slow API or `time.sleep`). Time both. This gives you a gut feeling for the distinction before any theory.

**Checkpoint questions:**
- What's the difference between concurrency and parallelism?
- Is downloading 100 files from the internet a CPU-bound or I/O-bound task? Why does that matter?
- In one sentence, what does the GIL prevent?

---

## Module 1: Synchronous Execution (the baseline you're escaping)

**Learn:**
- How a normal single-threaded Python script executes, line by line
- Why a blocking call (e.g. `requests.get()`) freezes the whole program until it returns
- The cost model: if you make 10 sequential blocking calls that take 1 second each, your program takes 10 seconds

**Hands-on:** Write a script that makes 5 sequential "slow" calls (`time.sleep(1)` is fine as a stand-in) and time it. This is your baseline to compare every later approach against.

**Checkpoint:** Why is synchronous code the natural default, and what specific problem does it cause at scale (e.g. a web server handling many users)?

---

## Module 2: Async Programming — Concepts First

This is the module where most self-taught devs get fuzzy. Go slow here.

**Learn, in this order:**
1. What a **coroutine** is (a function that can pause and resume — `async def`)
2. What **`await`** actually does: it doesn't "wait" in the blocking sense — it yields control back to the event loop
3. What the **event loop** is: a single-threaded loop that keeps a list of tasks, runs one until it hits an `await`, parks it, and picks up the next ready task
4. Why this only helps I/O-bound work, not CPU-bound work (the loop is still single-threaded — nothing runs *in parallel*, it just avoids sitting idle)
5. The difference between **concurrency** (async gives you) and **parallelism** (async does NOT give you)
6. What happens if you call a *blocking* function inside an `async def` — it blocks the entire event loop, freezing every other coroutine
7. `asyncio.create_task()` vs `await` directly — starting something "in the background" vs waiting for it immediately
8. `asyncio.gather()` — running multiple coroutines concurrently and collecting results
9. Async context managers (`async with`) and async iterators (`async for`) — where and why they exist (e.g. async DB connections)

**Hands-on:**
- Rewrite Module 1's 5-slow-calls script using `asyncio` and `asyncio.sleep`. Time it — it should now take ~1 second instead of 5.
- Break it on purpose: replace `asyncio.sleep` with real `time.sleep` inside one coroutine and watch everything else freeze. This is the single most illuminating exercise for understanding the event loop.
- Predict the print order of a script with 3 coroutines that `await asyncio.sleep()` for different durations, *before* running it.

**Checkpoint questions:**
- What is the event loop, in your own words, no jargon?
- Why does `await asyncio.sleep(1)` not block the whole program, but `time.sleep(1)` inside an `async def` does?
- If you have 100 I/O-bound tasks, will async make them faster than one thread doing them sequentially? Why?
- Will async make 100 CPU-bound tasks faster? Why not?
- What's the difference between `await coro()` and `asyncio.create_task(coro())`?

---

## Module 3: Multi-threading

**Learn:**
- What `threading.Thread` gives you vs asyncio: real OS-level threads, not coroutines
- Why threading *does* help I/O-bound work (a blocked thread releases the GIL, letting another thread run)
- Why threading does **NOT** help CPU-bound work in CPython (the GIL means only one thread executes Python bytecode at a time)
- **Race conditions**: what happens when two threads read-modify-write shared state without coordination
- **Locks** (`threading.Lock`): how they prevent race conditions, and the cost (contention, potential deadlock)
- **Deadlocks**: how two threads waiting on each other's locks freeze forever
- `threading.Event`, `Queue` (thread-safe) as coordination tools
- `concurrent.futures.ThreadPoolExecutor` — the higher-level, preferred way to use threads in practice

**Hands-on:**
- Write a race condition on purpose: spin up 10 threads that all increment a shared counter without a lock, run it multiple times, and watch the final value be wrong and inconsistent.
- Fix it with a `Lock`, confirm it's now consistent.
- Build a deadlock on purpose with two locks acquired in opposite order in two threads.
- Use `ThreadPoolExecutor` to fetch 20 URLs concurrently and compare timing to the sequential version.

**Checkpoint questions:**
- Why doesn't the GIL prevent race conditions, if only one thread runs Python bytecode at a time?
- What's a minimal example of a race condition, and how does a Lock fix it?
- What causes a deadlock, and how do you avoid it (e.g. lock ordering)?
- When would you choose threading over asyncio for I/O-bound work?

---

## Module 4: Multiprocessing

**Learn:**
- What `multiprocessing.Process` gives you: separate memory space, separate Python interpreter, separate GIL — true parallelism on multiple CPU cores
- Why this is the right tool for CPU-bound work (unlike threading)
- The cost: processes are heavier to start, and they **don't share memory** — you need explicit IPC (`Queue`, `Pipe`, shared memory) to pass data
- Serialization (`pickle`) — why data passed between processes must be picklable, and what that costs you
- `concurrent.futures.ProcessPoolExecutor` — the higher-level API most people actually use
- `multiprocessing.Pool.map()` for parallelizing a function over a list of inputs

**Hands-on:**
- Take your CPU-bound script (Module 0) and parallelize it with `ProcessPoolExecutor` across your CPU cores. Compare timing to single-process.
- Try (and watch fail) passing an unpicklable object (e.g. an open file handle, or a lambda) between processes — see the actual error.

**Checkpoint questions:**
- Why does multiprocessing bypass the GIL limitation, when multithreading doesn't?
- Why is IPC needed for multiprocessing but not multithreading?
- Given a CPU-bound task and an I/O-bound task, which tool (threading vs multiprocessing vs asyncio) fits each, and why?

---

## Module 5: The Decision Framework (tie it together)

By now you've built the same category of problem 3 different ways. This module is about internalizing *when* to use which.

**Learn / build yourself a mental table:**

| Task type | Best tool | Why |
|---|---|---|
| I/O-bound, many concurrent ops, want lightest weight | `asyncio` | No thread/process overhead, single loop handles thousands of tasks |
| I/O-bound, working with a library that isn't async-native | `threading` (ThreadPoolExecutor) | Threads release the GIL during I/O waits |
| CPU-bound | `multiprocessing` (ProcessPoolExecutor) | True parallelism across cores, bypasses the GIL |
| Mixed (CPU-bound work inside an async app) | `asyncio` + `run_in_executor` to offload to a process pool | Keeps the event loop responsive while heavy work runs elsewhere |

**Hands-on:** Build one script that has both an I/O-bound step and a CPU-bound step, and correctly offload the CPU part using `loop.run_in_executor()` with a `ProcessPoolExecutor`, while the I/O part stays `async`.

**Checkpoint:** Given any code snippet, can you say "this should be async / threaded / multiprocessed" and justify it in one sentence?

---

## Module 6: FastAPI — Now It All Clicks

This is where the three concepts meet a real framework, and previously confusing FastAPI behavior will suddenly make sense.

**Learn:**
- Why FastAPI is built on `async` (ASGI, not WSGI) — it's designed to handle many concurrent I/O-bound requests on one process without blocking
- `async def` routes vs plain `def` routes in FastAPI: `def` routes are automatically run in a thread pool by Starlette so they don't block the event loop — this is a huge "aha" once you understand Module 2 and 3
- Why calling a blocking library (e.g. a sync DB driver) inside an `async def` route is a classic bug that freezes your whole server under load
- Dependency injection (`Depends`) and how it interacts with async
- `BackgroundTasks` — for fire-and-forget work after a response is sent
- Async database drivers (e.g. `asyncpg`, async SQLAlchemy) vs sync ones, and why the choice matters
- Where multiprocessing/worker processes fit: Uvicorn/Gunicorn workers give you multiple *processes* (true parallelism across cores) while each process runs its own async event loop for concurrency within it — this is the "processes for parallelism, async for concurrency" pattern at the framework level
- Basic understanding of Uvicorn workers vs threads vs the event loop, so you can answer "how does FastAPI scale?"

**Hands-on:**
- Build a FastAPI app with one `async def` route that does real async I/O (e.g. `httpx.AsyncClient` calling an external API) and one `def` route doing the same work with a blocking library. Load test both with something like `wrk` or simple concurrent `curl`s, and observe the difference.
- Deliberately break it: put a `time.sleep()` inside an `async def` route, hit it with concurrent requests, and watch every other request stall.
- Add a CPU-bound task (e.g. image processing) to a route and fix the blocking event loop problem using a `ProcessPoolExecutor` via `run_in_executor`.
- Add a `BackgroundTasks` example (e.g. send an email after responding to the user).

**Checkpoint questions:**
- Why does FastAPI let you write both `async def` and `def` routes, and what does it do differently with each?
- What happens if you put a blocking call inside an `async def` route, under concurrent load?
- How would you handle a CPU-heavy task in a FastAPI route without blocking other requests?
- How do Uvicorn workers relate to multiprocessing, and how does that relate to what you learned about the GIL?

---

## Module 7: Production-Level Gotchas (the "no loose ends" layer)

**Learn:**
- Cancellation and timeouts in asyncio (`asyncio.wait_for`, task cancellation, cleanup with `try/finally`)
- Exception handling in concurrent code: an exception in one `asyncio.gather()` task vs one thread vs one process behaves differently — know each
- Thread/process pool sizing: why "more workers" isn't always faster (I/O-bound vs CPU-bound sizing rules of thumb)
- Shared state pitfalls in async code (yes, race conditions can still happen in async if you `await` between a read and a write on shared state)
- Graceful shutdown: closing DB connections, cancelling pending tasks
- Monitoring/debugging: how to tell if your event loop is blocked in production (e.g. `asyncio` debug mode, slow callback warnings)

**Hands-on:** Add a timeout and cancellation-safe cleanup to one of your earlier async examples. Reproduce a race condition in async code (two coroutines that `await` between reading and writing shared state).

**Checkpoint:** Can a race condition happen in single-threaded async code? Why or why not, and under what condition?

---

## Module 8: Interview Drill Mode

By now you've built and broken everything. This module is pure retrieval practice.

- Collect 15–20 "spot the bug" snippets covering: missing `await`, blocking call in async route, race condition without lock, deadlock, unpicklable object in multiprocessing, wrong tool for the job (e.g. threading for CPU-bound)
- Practice explaining each concept in under 30 seconds, out loud, no notes: event loop, GIL, coroutine, race condition, deadlock, process vs thread, async vs threading vs multiprocessing decision rule, `async def` vs `def` in FastAPI
- Do a mock "explain your project" pass: pick your FastAPI project and narrate exactly where you used async, where you used a thread/process pool, and why — this is almost always asked in interviews for projects like this

---

### How to use this roadmap
Don't move to the next module until you can answer that module's checkpoint questions without looking anything up. If you get stuck on a checkpoint, that's not a sign to skip it — it's a sign you found the actual gap. Go back into that module's hands-on exercise and poke at it more.