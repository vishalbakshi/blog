import os

def do_exit():
    if os.fork():
        # Parent process
        print("Parent waiting...")
        child_pid, status = os.waitpid(-1, 0)
        print("Parent done waiting!")
        print(f"In parent: {os.getpid()}")
        print(f"Child process (PID {child_pid}) exited with status {os.WEXITSTATUS(status)}")
    else:
        # Child process
        print(f"In child: {os.getpid()}, exiting with status 5")
        os._exit(5)  # Use os._exit to avoid affecting the notebook process

    print(f"This will be printed only by the parent process. PID: {os.getpid()}")

do_exit()