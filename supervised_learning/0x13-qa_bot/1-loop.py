#!/usr/bin/env python3
import sys


if __name__ == "__main__":
    command = ['exit', 'quit', 'goodbye', 'bye']
    while(True):
        argv = input("Q: ")
        if argv.strip() in command:
            print("A: Goodbye")
            break
        print("A: ")
