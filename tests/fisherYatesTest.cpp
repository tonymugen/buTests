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

#include <vector>
#include <chrono>
#include <iostream>

#include "random.hpp"

int main(){
	BayesicSpace::RanDraw tstRun;
	std::vector<uint64_t> res;
	std::chrono::duration<float, std::milli> upTime;
	std::chrono::duration<float, std::milli> downTime;
	auto time1 = std::chrono::high_resolution_clock::now();
	for (size_t l = 0; l < 50000; ++l){
		std::vector<uint64_t> loc{tstRun.shuffleUintUp(1000)};
		res.emplace_back(loc[0]);
	}
	auto time2 = std::chrono::high_resolution_clock::now();
	upTime = time2 - time1;
	time1 = std::chrono::high_resolution_clock::now();
	for (size_t l = 0; l < 50000; ++l){
		std::vector<uint64_t> loc{tstRun.shuffleUintDown(1000)};
		res.emplace_back(loc[0]);
	}
	time2 = std::chrono::high_resolution_clock::now();
	downTime = time2 - time1;
	std::cout << upTime.count() << "\tUp1000\tyes\n" << downTime.count() << "\tDown1000\tyes\n";
	return 0;
}
