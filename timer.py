import time


class Timer:
  def __init__(self):
    self.start_time = None
    self.note = None

  def start(self, note):
    self.start_time = time.time()
    self.note = note

  def check(self, var):
    tm = time.time()

    execution_time = tm - self.start_time
    print(f"Process '{self.note}' time check {var}: {execution_time:.5f} seconds")

  def stop(self):
    tm = time.time()

    execution_time = tm - self.start_time
    print(f"Process '{self.note}' executed in {execution_time:.5f} seconds")

    self.start_time = None
    self.note = None
