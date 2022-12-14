/*
 * Copyright (c) 2022 Anthony J. Greenberg
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
 * IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <array>
#include <iostream>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <random>
#include <algorithm>

#include "random.hpp"

#define NN 312
#define MM 156
#define MATRIX_A 0xB5026F5AA96619E9ULL
#define UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define LM 0x7FFFFFFFULL /* Least significant 31 bits */

uint64_t ranIntLooped() noexcept {
	unsigned long long rInt;
	// this can be in infinite loop if the random number pool is depleted
	// but then we have more serious system-wide issues
	while (_rdrand64_step(&rInt) == 0){
	}

	return static_cast<uint64_t> (rInt);
}

int32_t ranIntUnchk(unsigned long long &rInt) noexcept {
	return _rdrand64_step(&rInt);
}

/* The array for the state vector */
static unsigned long long mt[NN]; 
/* mti==NN+1 means mt[NN] is not initialized */
static unsigned long long mti=NN+1; 

/* initializes mt[NN] with a seed */
void init_genrand64(unsigned long long seed)
{
    mt[0] = seed;
    for (mti=1; mti<NN; mti++) 
        mt[mti] =  (6364136223846793005ULL * (mt[mti-1] ^ (mt[mti-1] >> 62)) + mti);
}

unsigned long long genrand64_int64(void)
{
    int i;
    unsigned long long x;
    static unsigned long long mag01[2]={0ULL, MATRIX_A};

    if (mti >= NN) { /* generate NN words at one time */

        /* if init_genrand64() has not been called, */
        /* a default initial seed is used     */
        if (mti == NN+1) 
            init_genrand64(5489ULL); 

        for (i=0;i<NN-MM;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+MM] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        for (;i<NN-1;i++) {
            x = (mt[i]&UM)|(mt[i+1]&LM);
            mt[i] = mt[i+(MM-NN)] ^ (x>>1) ^ mag01[(int)(x&1ULL)];
        }
        x = (mt[NN-1]&UM)|(mt[0]&LM);
        mt[NN-1] = mt[MM-1] ^ (x>>1) ^ mag01[(int)(x&1ULL)];

        mti = 0;
    }
  
    x = mt[mti++];

    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);

    return x;
}

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

//static uint64_t s[4];
static std::array<uint64_t, 4> s;

uint64_t next(void) {
	const uint64_t result = rotl(s[0] + s[3], 23) + s[0];

	const uint64_t t = s[1] << 17;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;

	s[3] = rotl(s[3], 45);

	return result;
}

int main(){
	std::chrono::duration<float, std::micro> xo256ppTime;
	std::chrono::duration<float, std::micro> mtTime;
	std::chrono::duration<float, std::micro> classTime;
	std::chrono::duration<float, std::micro> stdTime;
	std::chrono::duration<float, std::micro> xo256ppTimeS;
	std::chrono::duration<float, std::micro> mtTimeS;
	std::chrono::duration<float, std::micro> classTimeS;
	std::chrono::duration<float, std::micro> stdTimeS;
	std::array<uint64_t, 1000> res;
	uint64_t x = __rdtsc();
	BayesicSpace::RanDraw tstRan(x);
	for (size_t si = 0; si < s.size(); ++si){
		x         += 0x9e3779b97f4a7c15;
		uint64_t z = x;
		z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		s[si]      = z;
	}
	std::cout << tstRan.ranInt() << " " << tstRan.ranInt() << " | " << next() << " " << next() << "\n";
	std::mt19937_64 stdGen;
	stdGen.seed( __rdtsc() );
	init_genrand64( __rdtsc() );
	/*
	for (size_t i = 0; i < 4; ++i){
		s[i] = __rdtsc();
	}
	*/
	auto time1 = std::chrono::high_resolution_clock::now();
	res[1] = genrand64_int64();
	auto time2 = std::chrono::high_resolution_clock::now();
	mtTimeS = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	res[1] = next();
	time2 = std::chrono::high_resolution_clock::now();
	xo256ppTimeS = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	res[1] = tstRan.ranInt();
	time2 = std::chrono::high_resolution_clock::now();
	classTimeS = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	res[1] = stdGen();
	time2 = std::chrono::high_resolution_clock::now();
	stdTimeS = time2 - time1;
	std::cout << "single\t" << mtTimeS.count() << "\t" << xo256ppTimeS.count() << "\t" << classTimeS.count() << "\t" << stdTimeS.count() << "\n";
	
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 1000; ++i){
		res[i] = genrand64_int64();
	}
	time2 = std::chrono::high_resolution_clock::now();
	mtTime     = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 1000; ++i){
		res[i] = next();
	}
	time2       = std::chrono::high_resolution_clock::now();
	xo256ppTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 1000; ++i){
		res[i] = tstRan.ranInt();
	}
	time2     = std::chrono::high_resolution_clock::now();
	classTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 1000; ++i){
		res[i] = stdGen();
	}
	time2   = std::chrono::high_resolution_clock::now();
	stdTime = time2 - time1;
	
	std::cout << "loop\t" << mtTime.count() / 1000.0 << "\t" << xo256ppTime.count() / 1000.0 << "\t" << classTime.count() / 1000.0 << "\t" << stdTime.count() / 1000.0 << "\n";
	uint64_t max = 0;
	uint64_t min = std::numeric_limits<uint64_t>::max();
	for (size_t i = 0; i < 10000000000ULL; ++i){
		//uint64_t r = tstRun.ranInt();
		uint64_t r = stdGen();
		max        = std::max(max, r);
		min        = std::min(min, r);
		/*
		if ( tstRun.ranInt() >= 2*4294967295ULL ){
			++sum32;
		}
		*/
	}
	std::cout << min << " " << max << "\n";
}
