

// Factorial of a number using recursion

function factorial(n) {
    if (n === 0) return 1;
    return n * factorial(n - 1);
}

console.log(factorial(5));



// Fibonacci series using recursion
function fib(n) {
    if (n < 2) return n;    // base cases: fib(0)=0, fib(1)=1
    return fib(n - 1) + fib(n - 2);
}

console.log(fib(6));


// Iterative (bottom-up) alternative (simple and memory-friendly)

function fibIter(n) {
    if (n < 2) return n;
    let a = 0, b = 1, next;
    for (let i = 2; i <= n; i++) {
        next = a + b;
        a = b;
        b = next;
    }
    return b;
}

console.log(fibIter(6));
