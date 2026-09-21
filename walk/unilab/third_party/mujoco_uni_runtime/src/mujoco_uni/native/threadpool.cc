// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "threadpool.h"

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#else
#include <cerrno>
#endif

// NOTE: upstream keeps `#include <absl/base/attributes.h>` here and
// ABSL_CONST_INIT on worker_id_; dropped to avoid the Abseil dependency.
namespace mujoco::python {

namespace {

// Pins the calling thread to cpu_id. Returns 0 on success, an errno-style
// error code otherwise. CPU pinning is only supported on Linux.
int PinCurrentThreadToCpu(int cpu_id) {
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu_id, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
  (void)cpu_id;
  return ENOTSUP;
#endif
}

// Returns the CPU the calling thread is currently running on, or -1 when
// the platform does not support the query.
int CurrentCpu() {
#ifdef __linux__
  return sched_getcpu();
#else
  return -1;
#endif
}

}  // namespace

thread_local int ThreadPool::worker_id_ = -1;

// ThreadPool constructor
ThreadPool::ThreadPool(int num_threads, std::vector<int> cpu_ids)
    : ctr_(0),
      cpu_ids_(std::move(cpu_ids)),
      observed_cpus_(num_threads, -1),
      startup_count_(0),
      pin_error_(0) {
  for (int i = 0; i < num_threads; i++) {
    threads_.push_back(std::thread(&ThreadPool::WorkerThread, this, i));
  }
  // Cold path: block until every worker has applied its requested CPU
  // affinity and recorded the CPU it observed at startup, so CpuIds /
  // ObservedCpuIds / PinError are final once the constructor returns.
  while (startup_count_.load(std::memory_order_acquire) < num_threads) {
    std::this_thread::yield();
  }
}

// ThreadPool destructor
ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(m_);
    for (int i = 0; i < threads_.size(); i++) {
      queue_.push(nullptr);
    }
    cv_in_.notify_all();
  }
  for (auto& thread : threads_) {
    thread.join();
  }
}

// ThreadPool scheduler
void ThreadPool::Schedule(std::function<void()> task) {
  std::unique_lock<std::mutex> lock(m_);
  queue_.push(std::move(task));
  cv_in_.notify_one();
}

// ThreadPool worker
void ThreadPool::WorkerThread(int i) {
  worker_id_ = i;
  if (!cpu_ids_.empty()) {
    int err = PinCurrentThreadToCpu(cpu_ids_[i]);
    if (err != 0) {
      int expected = 0;
      pin_error_.compare_exchange_strong(expected, err,
                                         std::memory_order_relaxed);
    } else {
      observed_cpus_[i] = CurrentCpu();
    }
  }
  startup_count_.fetch_add(1, std::memory_order_release);
  while (true) {
    auto task = [&]() {
      std::unique_lock<std::mutex> lock(m_);
      cv_in_.wait(lock, [&]() { return !queue_.empty(); });
      std::function<void()> task = std::move(queue_.front());
      queue_.pop();
      cv_in_.notify_one();
      return task;
    }();
    if (task == nullptr) {
      {
        std::unique_lock<std::mutex> lock(m_);
        ++ctr_;
        cv_ext_.notify_one();
      }
      break;
    }
    task();

    {
      std::unique_lock<std::mutex> lock(m_);
      ++ctr_;
      cv_ext_.notify_one();
    }
  }
}

}  // namespace mujoco::python
