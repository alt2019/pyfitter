import logging

# Define some ANSI escape sequences for different colors.
RESET = "\x1b[0m"
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = [
  "\x1b[1;%dm" % (30 + i) for i in range(8)
]

# Define some colors for different logging levels.
colors = {
  "DEBUG": BLUE,
  "INFO": WHITE,
  "WARNING": YELLOW,
  "ERROR": RED,
  "CRITICAL": MAGENTA
}

# Create a custom formatter class that inherits from logging.Formatter.
class CustomFormatter(logging.Formatter):

  def __init__(self, fmt="%(levelname)s:%(name)s:%(message)s"):
    # Call the __init__() method of the parent class with the format string argument.
    super().__init__(fmt)
    # Set a custom attribute for the format string.
    self.fmt = fmt

  # Override the format() method to add colors to the logging messages.
  def format(self, record):
    # print("RECORD:", record, type(record))
    record.levelname = f"{colors.get(record.levelname, WHITE)}{record.levelname}{RESET}"
    record.name = f"{GREEN}{record.name}{RESET}"
    record.funcName = f"{CYAN}{record.funcName}{RESET}"
    record.lineno = f"{YELLOW}{record.lineno}{RESET}"

    # Get the original message from the record object.
    message = super().format(record)

    # print("message:", message)

    # Get the color for the current logging level.
    # color = colors.get(record.levelname, WHITE)
    # # Add the color and reset sequences to the message.
    # colored_message = color + message + RESET
    # Return the colored message as the new formatted message.
    # return colored_message
    return message



# Create a custom logger class that inherits from logging.Logger.
class CustomLogger(logging.Logger):

  # Override the __init__() method to set up some default attributes and handlers.
  def __init__(self, name):
    # Call the __init__() method of the parent class.
    super().__init__(name)
    # Create a handler object that prints to the standard output stream.
    handler = logging.StreamHandler()
    # Create a formatter object that formats the logging messages.
    # formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    fmt = "[%(name)s:%(funcName)s:%(lineno)s]:%(levelname)s: %(message)s"
    formatter = CustomFormatter(fmt)

    # Set the formatter for the handler.
    handler.setFormatter(formatter)
    # Add the handler to the logger.
    self.addHandler(handler)

  # Add a custom method that logs a greeting message.
  def greet(self, name):
    self.info(f"Hello, {name}!")
