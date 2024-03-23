#!/usr/bin/env python3
# Description: Build script for the project
import subprocess
import os

GREEN 	= "\u001b[32m"
RED 	= "\u001b[31m"
YELLOW 	= "\u001b[33m"
CLEAR 	= "\u001b[0m"

ENV = os.environ.copy()
SCRIPT_ENV_VARS = []

def run(command, output: bool = False, include_env: bool = False):
	result = subprocess.run(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=ENV if include_env else None)
	# check if the command was successful
	if result.returncode != 0:
		raise Exception(f"{RED}\nBUILD - Command failed: {' '.join(command)}{CLEAR}\nReason:\n{result.stdout.decode('utf-8')}")

	if output:
		print(GREEN + f"BUILD - Command output:\n{CLEAR}{result.stdout.decode('utf-8')}")


def set_env_var(name: str, value: str):
	ENV[name] = value
	SCRIPT_ENV_VARS.append(name)
	
def main():
	# set environment variables
	set_env_var("CC", "clang")
	set_env_var("CXX", "clang++")

	def meson():
		print(f"{GREEN}BUILD - Environment variables:{CLEAR}")
		for env_var in SCRIPT_ENV_VARS: print(f"      - {env_var}={ENV[env_var]}")

		# if build directory exists, delete it
		if os.path.exists("build"):
			run(["rm", "-rf", "build"])

		command = ["meson", "setup", "build"]

		print(f"{GREEN}BUILD - Meson command: {CLEAR}" + " ".join(command))
		run(command, output=True, include_env=True)

	def ninja():
		command = ["ninja", "-C", "build"]
		print(f"{GREEN}BUILD - Ninja command: {CLEAR}" + " ".join(command))
		run(command, output=True)

	def clean():
		print(f"{GREEN}BUILD - Cleaning build directory{CLEAR}")
		run(["rm", "-rf", "build"])

	import argparse
	parser = argparse.ArgumentParser(description="Build script for the project")
	parser.add_argument("-m", "--meson", action="store_true", help="Run meson")
	parser.add_argument("-n", "--ninja", action="store_true", help="Run ninja")
	parser.add_argument("-c", "--clean", action="store_true", help="Clean build directory")
	args = parser.parse_args()

	# validate arguments
	if not args.meson and not args.ninja and not args.clean:
		print(f"{RED}BUILD - No arguments provided{CLEAR}")
		parser.print_help()
		exit(1)

	if args.meson:
		meson()
	if args.ninja:
		ninja()
	if args.clean:
		clean()

	print(f"{YELLOW}BUILD - BUILD SCRIPT FINISHED{CLEAR}")

if __name__ == "__main__":
	main()