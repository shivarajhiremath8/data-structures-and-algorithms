
// Note: Big O is used to measure the performance of an algorithm. It tells us how the runtime or space requirements grow as the input size grows.

// -------------------------------  ------------------------------- //

// O(n)

// Print 1 to N
function printOneToN(n) {
    for (let i = 1; i <= n; i++) {
        console.log(i)
    }
    console.log("DONE")
}

printOneToN(3)

// Print the sum of 1 to N
function sum(n) {
    let sum = 0
    for (let num = 1; num <= n; num++) {
        sum += num
    }
    return sum
}

console.log(sum(3)) // output: 6

// Time complexity: O(n)

// -------------------------------  ------------------------------- //

// O(n^2)

// Print all pairs from 1 to N
function printPairs(n) {
    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= n; j++) {
            console.log(`${i}-${j}`)
        }
    }
    console.log("DONE")
}

printPairs(3)

// Print multiplication table from 0 to N
function printMultiplicationTable(n) {
    for (let a = 0; a <= n; a++) {
        for (let b = 0; b <= n; b++) {
            console.log(`${a} x ${b} = ${a * b}`)
        }
    }
}
printMultiplicationTable(3) // output: 0x0=0, 0x1=0, 0x2=0, 0x3=0, 1x0=0, 1x1=1, 1x2=2, 1x3=3, 2x0=0, 2x1=2, 2x2=4, 2x3=6, 3x0=0, 3x1=3, 3x2=6, 3x3=9

// Time complexity: O(n^2)

// -------------------------------  ------------------------------- //

// O(1)

// Print a message with the number of dinosaurs in our group
function printGroupMessage(n) {
    console.log(`There are ${n} dinosaurs in our group.`)
}

printGroupMessage(3)

// Check if a number is positive
function isPositive(n) {
    return n > 0
}
// Time complexity: O(1)

// Print a triangle of stars
function printTriangle() {
    for (let row = 1; row <= 5; row++) {
        let line = ""
        for (let col = 1; col <= row; col++) {
            line = line + "*"
        }
        console.log(line)
    }
}
printTriangle()

// Time complexity: O(1)

// -------------------------------  ------------------------------- //

// O(log n)


// -------------------------------  ------------------------------- //

// O(n log n)

// -------------------------------  ------------------------------- //
