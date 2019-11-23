#!/usr/bin/expect -f
set rdir [lindex $argv 0]

spawn bash -c "scp -r z1123375@student.tcs.uj.edu.pl:/home/z1123375/afs/profil_TCS/Desktop/cudaLic/* ./fromStudent"
expect "*?assword:*"
send -- "g11h06dr59\r"
send -- "\r"
expect eof
