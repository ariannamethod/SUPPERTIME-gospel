"""Legacy entry point maintained for backward compatibility."""
from bridge import *  # noqa: F401,F403
from theatre import *  # noqa: F401,F403
from db import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
