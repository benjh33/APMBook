#! /bin/bash
sudo apt-get -y install software-properties-common
sudo add-apt-repository ppa:george-edison55/cmake-3.x
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9  
sudo add-apt-repository ppa:marutter/rdev
sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install git build-essential python-dev libsqlite3-dev libreadline6-dev libgdbm-dev zlib1g-dev libbz2-dev sqlite3 zip python3 python3-dev python3-pip cmake r-base r-base-dev default-jre libssl-dev

git clone git://github.com/ansible/ansible.git --recursive
git clone https://github.com/benjh33/vim --recursive .vim

sudo apt-get -y install gdebi-core
wget https://download2.rstudio.org/rstudio-server-0.99.902-amd64.deb
sudo gdebi rstudio-server-0.99.902-amd64.deb
