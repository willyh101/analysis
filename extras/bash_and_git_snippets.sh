# set VSCode as defaule git editor
git config --global core.editor "code --wait"

# pull changes from local branch
git merge master

# configure remote fork
git remote -v
git remote add upstream https://github.com/MouseLand/suite2p
git remote -v

# fetch from upstream remote
git fetch upstream
git checkout main
git merge upstream/main

# stash changes
git stash
git stash -u # incl. untracked

# manual merging
# on branch you want to merge changes into
git merge dev
# if there are conflicts...
# 1. fix them in editor
# 2. if there are unmerged paths...
git status # find modified file
git add file.py # or rm file if needed
git commit
git push

# setup bare bones remote git server
git init --bare ~/projectname.git
# then on local pc...
cd /path/to/local/project
git init .
git remote add origin user@git_server:projectname.git
git add .
git commit -m "initial commit"
git push -u origin main