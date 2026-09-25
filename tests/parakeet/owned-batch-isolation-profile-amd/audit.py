"""Run the unchanged complete-request, resource and phase/node audit."""
from run import ORIGINAL, load

original = load('retained_profile_audit', ORIGINAL/'audit.py')


if __name__ == '__main__':
    original.main()
