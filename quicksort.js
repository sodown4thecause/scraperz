/**
 * Quicksort algorithm implementation in JavaScript
 * Time Complexity: O(n log n) average case, O(n²) worst case
 * Space Complexity: O(log n) due to recursion stack
 * @param {Array} arr - The array to be sorted
 * @param {number} left - Left index (default: 0)
 * @param {number} right - Right index (default: arr.length - 1)
 * @returns {Array} - The sorted array
 */
function quicksort(arr, left = 0, right = arr.length - 1) {
    // Base case: if the array has 1 or 0 elements, it's already sorted
    if (left >= right) {
        return arr;
    }

    // Partition the array and get the pivot index
    const pivotIndex = partition(arr, left, right);

    // Recursively sort the left and right subarrays
    quicksort(arr, left, pivotIndex - 1);
    quicksort(arr, pivotIndex + 1, right);

    return arr;
}

/**
 * Partition helper function for quicksort
 * @param {Array} arr - The array to partition
 * @param {number} left - Left index
 * @param {number} right - Right index
 * @returns {number} - The final position of the pivot
 */
function partition(arr, left, right) {
    // Choose the rightmost element as pivot
    const pivot = arr[right];
    let i = left - 1;

    // Iterate through the array and rearrange elements
    for (let j = left; j < right; j++) {
        if (arr[j] <= pivot) {
            i++;
            // Swap arr[i] and arr[j]
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }
    }

    // Place the pivot in its correct position
    [arr[i + 1], arr[right]] = [arr[right], arr[i + 1]];

    return i + 1;
}

// Example usage and test cases
function testQuicksort() {
    console.log("Testing Quicksort Implementation:");

    // Test case 1: Basic array
    const arr1 = [64, 34, 25, 12, 22, 11, 90];
    console.log("Original:", arr1);
    quicksort(arr1);
    console.log("Sorted:", arr1);

    // Test case 2: Already sorted array
    const arr2 = [1, 2, 3, 4, 5];
    console.log("Original:", arr2);
    quicksort(arr2);
    console.log("Sorted:", arr2);

    // Test case 3: Reverse sorted array
    const arr3 = [5, 4, 3, 2, 1];
    console.log("Original:", arr3);
    quicksort(arr3);
    console.log("Sorted:", arr3);

    // Test case 4: Array with duplicates
    const arr4 = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
    console.log("Original:", arr4);
    quicksort(arr4);
    console.log("Sorted:", arr4);

    // Test case 5: Single element
    const arr5 = [42];
    console.log("Original:", arr5);
    quicksort(arr5);
    console.log("Sorted:", arr5);
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { quicksort, partition };
}

// Run tests if this file is executed directly
if (typeof require !== 'undefined' && require.main === module) {
    testQuicksort();
}
