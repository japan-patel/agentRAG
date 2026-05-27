# Python Async Programming

## Introduction

Async programming in Python allows you to write concurrent code using the async/await syntax.

## Basic Example

Here's how to define an async function:

```python
import asyncio

async def main():
    print("Hello")
    await asyncio.sleep(1)
    print("World")

asyncio.run(main())
```

## Key Concepts

- `async def` defines an async function
- `await` pauses execution until the awaited task completes
- `asyncio.run()` is the entry point for async programs