from pyspark import SparkContext, SparkConf
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import sys
import random
import numpy as np

# After how many items should we stop?
THRESHOLD = 1e6
p = 8191


def update_count_sketch(item, count_sketch, hash_function_values, sign_hash_function_values, D, W):
    for row in range(D):
        hash_a, hash_b = hash_function_values[row]
        column = ((hash_a * item + hash_b) % p) % W

        sign_a, sign_b = sign_hash_function_values[row]

        sign = -1 if ((sign_a * item + sign_b) & 1) else 1

        count_sketch[row][column] += sign


# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch):
    # We are working on the batch at time `time`.
    global streamLength, histogram, count_sketch, F2, approximation_f2, hash_function_values, D, W, left, right

    batch_size = batch.count()
    # If we already have enough points (> THRESHOLD), skip this batch.
    if streamLength[0] >= THRESHOLD:
        return

    streamLength[0] += batch_size
    # Extract the distinct items from the batch
    batch_items = batch.map(lambda s: int(s)).filter(lambda x: left <= x <= right).collect()
    filtered_stream_length[0] += len(batch_items)

    for item in batch_items:
        if item not in histogram:
            histogram[item] = 1
        else:
            histogram[item] += 1

        update_count_sketch(item, count_sketch, hash_function_values, sign_hash_function_values, D, W)

    if streamLength[0] >= THRESHOLD:
        stopping_condition.set()


def generate_hash_function(D):
    hash_function_values = [(random.randint(1, p - 1), random.randint(0, p - 1)) for _ in range(D)]
    sign_hash_function_values = [(random.randint(1, p - 1), random.randint(0, p - 1)) for _ in range(D)]
    return hash_function_values, sign_hash_function_values


def count_sketch_estimate(item, count_sketch, hash_function_values, sign_hash_function_values, D, W):
    estimates = []
    for row in range(D):
        hash_a, hash_b = hash_function_values[row]
        column = ((hash_a * item + hash_b) % p) % W

        sign_a, sign_b = sign_hash_function_values[row]

        sign = -1 if ((sign_a * item + sign_b) & 1) else 1

        estimate = count_sketch[row][column] * sign
        estimates.append(estimate)

    return int(np.median(estimates))


# This code calculates approximate frequencies of the top-K items in a streaming data using the CountSketch algorithm.

# Check if the code is being executed as the main module
if __name__ == '__main__':
    assert len(sys.argv) == 7  # Ensure that the required number of command-line arguments are provided

    # Set up Spark configuration
    conf = SparkConf().setMaster("local[*]").setAppName("CountSketch").set("spark.executor.memory", "4g").set(
        "spark.driver.memory", "4g")

    sc = SparkContext(conf=conf)  # Create a SparkContext
    ssc = StreamingContext(sc, 1)  # Create a StreamingContext with a batch interval of 1 second
    ssc.sparkContext.setLogLevel("ERROR")  # Set the log level to ERROR

    stopping_condition = threading.Event()  # Create a threading event for stopping the streaming context

    # Extract the command-line arguments
    D, W, left, right, K, portExp = map(int, sys.argv[1:])

    streamLength = [0]  # Initialize a list to hold the total number of items in the stream
    filtered_stream_length = [
        0]  # Initialize a list to hold the total number of items in the specified range [left, right]
    histogram = {}  # Initialize an empty dictionary to store item frequencies
    F2 = 0  # Initialize F2 score
    approximation_f2 = 0  # Initialize approximate F2 score
    count_sketch = [[0] * W for _ in range(D)]  # Initialize the CountSketch data structure

    hash_function_values, sign_hash_function_values = generate_hash_function(D)  # Generate hash function values

    # Create a socket stream to read data from 'algo.dei.unipd.it' on the specified port
    stream = ssc.socketTextStream('algo.dei.unipd.it', portExp, StorageLevel.MEMORY_AND_DISK)

    # Process each batch of data using the process_batch function
    stream.foreachRDD(lambda time, batch: process_batch(time, batch))

    ssc.start()  # Start the streaming context
    stopping_condition.wait()  # Wait for the stopping condition to be triggered
    ssc.stop(False, True)  # Stop the streaming context gracefully

    # Print the parameters used
    print(f"D = {D} W = {W} [left,right] = [{left},{right}] K = {K} Port = {portExp}")

    # Print the results
    print("Total number of items =", streamLength[0])
    print(f"Total number of items in [{left},{right}] =", filtered_stream_length[0])
    print(f"Number of distinct items in [{left},{right}] =", len(histogram))

    # Get the top-K items based on their frequencies
    top_K_items = sorted(histogram, key=histogram.get, reverse=True)[:K]

    # Calculate the exact frequencies of the top-K items
    exact_frequencies = {item: histogram[item] for item in top_K_items}

    # Calculate the approximate frequencies of the top-K items using the CountSketch algorithm
    approx_frequencies = {
        item: count_sketch_estimate(item, count_sketch, hash_function_values, sign_hash_function_values, D, W) for
        item in top_K_items}

    # Calculate the errors between the exact and approximate frequencies
    errors = [abs(exact_frequencies[item] - approx_frequencies[item]) / exact_frequencies[item] for item in top_K_items]

    # Calculate the average relative error
    avg_relative_error = sum(errors) / len(errors)

    # Calculate the F2 score
    F2 = sum([histogram[item] ** 2 for item in histogram])

    # Calculate the approximate F2 score
    approximation_f2 = sum([approx_frequencies[item] ** 2 for item in approx_frequencies])

    if K <= 20:
        # Print the exact and approximate frequencies for the top-K items
        for item in top_K_items:
            print(f"Item {item} Freq = {exact_frequencies[item]} Est. Freq = {approx_frequencies[item]}")

    # Print the average error for the top-K items
    print(f"Avg err for top {K} = {avg_relative_error}")

    # Print the F2 score and the approximate F2 score normalized by the square of the number of items in the specified range
    print(
        f"F2 {F2 / (filtered_stream_length[0] ** 2)} F2 Estimate {approximation_f2 / (filtered_stream_length[0] ** 2)}")
