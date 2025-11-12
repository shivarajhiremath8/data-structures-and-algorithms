
// -------------------------------  ------------------------------- //


// Check if a number is even or odd
let num = 7;
if (num % 2 === 0) {
    console.log("Even");
} else {
    console.log("Odd");
}


// -------------------------------  ------------------------------- //


// Find the largest of three numbers
let a = 10, b = 25, c = 15;
if (a >= b && a >= c) console.log(a);
else if (b >= a && b >= c) console.log(b);
else console.log(c);


// -------------------------------  ------------------------------- //


// Print the sum of 1 to N
let n = 10, sum = 0;
for (let i = 1; i <= n; i++) sum += i;
console.log(sum);


// -------------------------------  ------------------------------- //


// Find the factorial of a number
let number = 5, fact = 1;
for (let i = 1; i <= number; i++) fact *= i;
console.log(fact);


// -------------------------------  ------------------------------- //

// Check if a number is prime
function isPrime(num) {
    if (num < 2) return false;
    for (let i = 2; i <= Math.sqrt(num); i++) {
        if (num % i === 0) return false;
    }
    return true;
}

console.log(isPrime(7)); // true
console.log(isPrime(10)); // false

// -------------------------------  ------------------------------- //


// Reverse a number
let num1 = 1234;
let rev = 0;
while (num1 > 0) {
    let rem = num1 % 10;
    rev = rev * 10 + rem;
    num1 = Math.floor(num1 / 10);
}
console.log(rev); // 4321


// -------------------------------  ------------------------------- //
