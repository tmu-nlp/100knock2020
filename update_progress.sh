#!/bin/bash

# initial configure
name="wkwkgg"
email="yujint7i@gmail.com"

# git configure
git config --global user.name $name
git config --global user.email $email

git remote set-url origin https://${name}:${GITHUB_TOKEN}@github.com/tmu-nlp/100knock2020.git

last_commit_msg="$(git log -1 | tail -1)"

if [ -z "$(echo $last_commit_msg | grep Bot)"]; then
    python3 make_progress.py
    git add progress.png
    git commit -m "[Bot] update progress bar"
    git push origin HEAD
fi
