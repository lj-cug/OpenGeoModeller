# Dependency check for the project
import subprocess

# Dependency list
DEPENDENCIES = [
	"python3",
	"pip",
	"meson",
	"ninja",
]

# Check if the dependency is installed
def check_dependency(dependency):
	try:
		subprocess.run([dependency, "--version"], shell=False, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		return True
	except:
		return False

# Check if all dependencies are installed
def check_dependencies():
	missing_dependencies = []

	# Check if all dependencies are installed
	for dependency in DEPENDENCIES:
		if not check_dependency(dependency):
			missing_dependencies.append(dependency)

	# Return the missing dependencies
	if len(missing_dependencies) > 0:
		return missing_dependencies
	else:
		return None


# Print the dependency check result
def print_dependency_check():
	result = check_dependencies()
	# Print the dependency check result
	if result is None:
		print("\u001b[32m" + "[+] All dependencies are installed")
	else:
		print("\u001b[30m" + "[-] One or more dependencies are missing!")
		for dependency in result:
			print("\u001b[30m" + f"    [-] Missing: {dependency}")

# Run the dependency check
if __name__ == "__main__":
	print_dependency_check()