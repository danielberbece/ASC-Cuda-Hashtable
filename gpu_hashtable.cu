#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <cmath>

#include "gpu_hashtable.hpp"


// Auxiliary methods
__device__ unsigned long long hashKey(int key, unsigned long long int capacity) {
	return ((175372756929481llu * (unsigned long long)key ) % 905035071625626043llu) % (unsigned long long int)capacity;
}

__device__ hashelem_t makeElement(int key, int value) {
   return (((hashelem_t) key) << 8 * sizeof(value)) | value;
}

__device__ int elementKey(hashelem_t e) {
   return e >> 8 * sizeof(int);
}

__device__ int elementValue(hashelem_t e) {
   return (int) (e & ((1ULL << 8 * sizeof(int))-1));
}

/* INIT HASH
 */
__global__ void kernel_init_table(hashtable_t *hashtable, hashelem_t *table, unsigned long long size) {
	hashtable->table = table;
	hashtable->capacity = size;
	hashtable->items = 0;
}

// Create Hashtable
GpuHashTable::GpuHashTable(int size) {
	// Allocate memory
	hashelem_t *table = NULL;
	cudaMalloc((void **) &table, size * sizeof(hashelem_t));
	cudaMemset(table, NULL_HASHELEM, size);
	cudaMalloc((void **) &deviceHashtable, sizeof(hashtable_t));

	// Initialize hashtable structure
	kernel_init_table<<<1, 1>>>(deviceHashtable, table, size);
	cudaDeviceSynchronize();

	arraySize = 1;
	allocArrays();
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	hashtable_t *hostHashtable = (hashtable_t *)calloc(1, sizeof(hashtable_t));
	cudaMemcpy(hostHashtable, deviceHashtable, sizeof(hashtable_t), cudaMemcpyDeviceToHost);
	cudaFree(hostHashtable->table);
	free(hostHashtable);
	// Destroy arrays for data distribution
	free(hostValues);
	cudaFree(deviceHashtable);
	cudaFree(deviceKeys);
	cudaFree(deviceValues);
}

// Allocate memory for data distribution arrays
void GpuHashTable::allocArrays() {
	hostValues = (int *) calloc(arraySize, sizeof(int));
	cudaMalloc((void **) &deviceKeys, arraySize * sizeof(int));
	cudaMalloc((void **) &deviceValues, arraySize * sizeof(int));
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int batchSize) {
	if (batchSize > arraySize) {
		arraySize = batchSize;
		cudaFree(deviceValues);
		cudaFree(deviceKeys);
		free(hostValues);
		allocArrays();		
	}
}

/* INSERT BATCH
 */
__global__ void kernel_rehash(hashtable_t *hashtable, hashelem_t *newTable, int newCapacity) {
	// Rehash old elements from hashtable and insert them to the newTable
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index > hashtable->capacity)
		return;
	
	// Get key and value for this thread
	hashelem_t e = hashtable->table[index];
	if (e == NULL_HASHELEM)	// If element is empty, we are done
		return;

	int key = elementKey(e);
	unsigned long long int hashValue = hashKey(key, newCapacity);
	unsigned long long int tableIndex = hashValue;
	// Try to add using quadratic probing:
	for (unsigned long long int i = 0; i < newCapacity; i++) {
		// Use atomicCAS for concurrency management
		hashelem_t currElem = atomicCAS(newTable + tableIndex, NULL_HASHELEM, e);
		if (currElem == NULL_HASHELEM) // If element was inserted in hashtable, return
			return;

		// If element wasn't inserted, get the next tableIndex using quadratic probing
		tableIndex = (hashValue + i*i) % newCapacity;
	}
}

__global__ void kernel_insert(hashtable_t *hashtable, int *keys, int *values, int numKeys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index > numKeys)
		return;

	// Get key and value for this thread
	int key = keys[index];
	int value = values[index];
	if (key == 0 || value == 0)	// If either key or value is 0, don't do anything
		return;

	unsigned long long int hashValue = hashKey(key, hashtable->capacity);
	unsigned long long int tableIndex = hashValue;
	
	// Make hash element
	hashelem_t e = makeElement(key, value);

	// Try to add using quadratic probing:
	for (unsigned long long int i = 0; i < hashtable->capacity; i++) {
		// Use atomicCAS for concurrency management
		hashelem_t currElem = atomicCAS(hashtable->table + tableIndex, NULL_HASHELEM, e);
		if (currElem == NULL_HASHELEM) { // If element was inserted in hashtable
			atomicAdd(&hashtable->items, 1);
			return;
		} else if (elementKey(currElem) == key) {	// If the currElem has the same key, then update
			atomicExch(hashtable->table + tableIndex, e);
			return;
		}

		// If element wasn't inserted, get the next tableIndex using quadratic probing
		tableIndex = (hashValue + i*i) % hashtable->capacity;
	}

	// Shouldn't get here unless the hash is full, but we checked for that in checkLoad()
}

void GpuHashTable::checkLoad(unsigned long long int batchSize) {
	// Compute the load factor
	hashtable_t *hostHashtable = (hashtable_t *)calloc(1, sizeof(hashtable_t));
	double loadFactor;
	
	cudaMemcpy(hostHashtable, deviceHashtable, sizeof(hashtable_t), cudaMemcpyDeviceToHost);
	loadFactor = ((double) hostHashtable->items + batchSize) / ((double) hostHashtable->capacity);

	// If load factor too big, increase the hash's capacity
	if (loadFactor > MAX_LOAD) {
		// Compute new capacity:
		unsigned long long int newCapacity = ((double)hostHashtable->items + (double)batchSize) / (double) MIN_LOAD;

		// Create new table
		hashelem_t *newTable = NULL;
		cudaMalloc((void **) &newTable, newCapacity * sizeof(hashelem_t));
		cudaMemset(newTable, NULL_HASHELEM, newCapacity);

		// Rehash elements in hashtable
		int numBlocks = hostHashtable->capacity / 512 + 1;
		kernel_rehash<<<numBlocks, 512>>>(deviceHashtable, newTable, newCapacity);
		cudaDeviceSynchronize();

		// Exchange the old hashtable with the new table
		cudaFree(hostHashtable->table);
		hostHashtable->table = newTable;
		hostHashtable->capacity = newCapacity;
		
		// Copy hashtable info from host to device
		cudaMemcpy(deviceHashtable, hostHashtable, sizeof(hashtable_t), cudaMemcpyHostToDevice);
	}

	free(hostHashtable);
}

bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	checkLoad(numKeys);
	reshape(numKeys);
	int numBlocks = numKeys / 512 + 1;

	// Copy data from host to device
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	kernel_insert<<<numBlocks, 512>>>(deviceHashtable, deviceKeys, deviceValues, numKeys);
	cudaDeviceSynchronize();

	return true;
}

/* GET BATCH
 */
__global__ void kernel_get_batch(hashtable_t *hashtable, int *keys, int *values, int numKeys) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index > numKeys)
		return;

	// Get key for this thread and try to find it's value in the hashtable
	int key = keys[index];
	unsigned long long int hashcode = hashKey(key, hashtable->capacity);
	unsigned long long int tableIndex = hashcode;
	hashelem_t elem;

	// Get hash element using quadratic probing:
	for(unsigned long long int i = 0; i < hashtable->capacity; i++) {
		elem = hashtable->table[tableIndex];
		if (elementKey(elem) == key) { // If element has the same key, save the value
			values[index] = elementValue(elem);
			return;
		}

		// If element wasn't found, get the next tableIndex using quadratic probing
		tableIndex = (hashcode + i*i) % hashtable->capacity;
	}
}

int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int numBlocks = numKeys / 512 + 1;

	// Copy keys array from host to device and start kernel
	cudaMemcpy(deviceKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	kernel_get_batch<<<numBlocks, 512>>>(deviceHashtable, deviceKeys, deviceValues, numKeys);
	cudaDeviceSynchronize();

	// Copy back the values from device to host and return them
	cudaMemcpy(hostValues, deviceValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

	return hostValues;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	// Get hashtable info from the device to host and compute load
	hashtable_t *hostHashtable = (hashtable_t *)calloc(1, sizeof(hashtable_t));
	float loadFactor;
	
	cudaMemcpy(hostHashtable, deviceHashtable, sizeof(hashtable_t), cudaMemcpyDeviceToHost);
	loadFactor = ((float) hostHashtable->items) / ((float) hostHashtable->capacity);

	free(hostHashtable);
	return loadFactor;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(100);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
