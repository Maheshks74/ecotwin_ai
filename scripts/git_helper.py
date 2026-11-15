"""Git helper using dulwich so git operations work cleanly across environments."""
import os
from dulwich import porcelain

REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def init_repo():
    git_dir = os.path.join(REPO_PATH, ".git")
    if not os.path.exists(git_dir):
        porcelain.init(REPO_PATH)
        print(f"Initialized git repo at {REPO_PATH}")
    else:
        print(f"Git repo already exists at {REPO_PATH}")

def commit_all(message: str, author: str = "EcoTwin <ecotwin@local.twin>"):
    init_repo()
    porcelain.add(REPO_PATH, paths=None)
    commit_id = porcelain.commit(
        REPO_PATH,
        message=message.encode("utf-8"),
        author=author.encode("utf-8"),
        committer=author.encode("utf-8")
    )
    print(f"Committed {commit_id.decode('utf-8')[:7]}: {message}")
    return commit_id

if __name__ == "__main__":
    init_repo()
    commit_all("Initial commit with environment and requirements")

