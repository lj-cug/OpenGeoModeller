git add .
git commit -m "update main branch"
git push origin master

cd hpc-base
git add .
git commit -m "update submodule hpc-base"
git push
cd ..

cd agent-dev
git add .
git commit -m "update submodule agent-dev"
git push
cd ..
