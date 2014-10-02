#!/bin/bash

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp1', 17999, 19999, [-0.1, 0.5], [0, 1.1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist80_0.exp1.particle.avi ./tmp/freq18k20k.dist80_0.exp1.particle.mp4 -y
rm ./tmp/freq18k20k.dist80_0.exp1.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp1', 17999, 19999, [-0.1, 0.5], [0, 1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist0_30.exp1.particle.avi ./tmp/freq18k20k.dist0_30.exp1.particle.mp4 -y
rm ./tmp/freq18k20k.dist0_30.exp1.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.pic1', 17998.9, 19999, [-0.1, 0.2], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic1.particle.avi ./tmp/freq18k20k.pic1.particle.mp4 -y
rm ./tmp/freq18k20k.pic1.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.pic2', 17999, 19999, [-0.1, 0.1], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic2.particle.avi ./tmp/freq18k20k.pic2.particle.mp4 -y
rm ./tmp/freq18k20k.pic2.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.pic3', 17999, 19999, [-0.05, 0.3], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic3.particle.avi ./tmp/freq18k20k.pic3.particle.mp4 -y
rm ./tmp/freq18k20k.pic3.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.pic4', 17999, 19999, [-0.1, 0.2], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic4.particle.avi ./tmp/freq18k20k.pic4.particle.mp4 -y
rm ./tmp/freq18k20k.pic4.particle.avi







/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp2', 17999, 19999, [-0.1, 0.5], [0, 1.1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist80_0.exp2.particle.avi ./tmp/freq18k20k.dist80_0.exp2.particle.mp4 -y
rm ./tmp/freq18k20k.dist80_0.exp2.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp2', 17999, 19999, [-0.1, 0.5], [0, 1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist0_30.exp2.particle.avi ./tmp/freq18k20k.dist0_30.exp2.particle.mp4 -y
rm ./tmp/freq18k20k.dist0_30.exp2.particle.avi


/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp3', 17999, 19999, [-0.1, 0.5], [0, 1.1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist80_0.exp3.particle.avi ./tmp/freq18k20k.dist80_0.exp3.particle.mp4 -y
rm ./tmp/freq18k20k.dist80_0.exp3.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp3', 17999, 19999, [-0.1, 0.5], [0, 1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist0_30.exp3.particle.avi ./tmp/freq18k20k.dist0_30.exp3.particle.mp4 -y
rm ./tmp/freq18k20k.dist0_30.exp3.particle.avi


/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp4', 17999, 19999, [-0.1, 0.5], [0, 1.1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist80_0.exp4.particle.avi ./tmp/freq18k20k.dist80_0.exp4.particle.mp4 -y
rm ./tmp/freq18k20k.dist80_0.exp4.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp4', 17999, 19999, [-0.1, 0.5], [0, 1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist0_30.exp4.particle.avi ./tmp/freq18k20k.dist0_30.exp4.particle.mp4 -y
rm ./tmp/freq18k20k.dist0_30.exp4.particle.avi


/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp5', 17999, 19999, [-0.1, 0.5], [0, 1.1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist80_0.exp5.particle.avi ./tmp/freq18k20k.dist80_0.exp5.particle.mp4 -y
rm ./tmp/freq18k20k.dist80_0.exp5.particle.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position_particle('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp5', 17999, 19999, [-0.1, 0.5], [0, 1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist0_30.exp5.particle.avi ./tmp/freq18k20k.dist0_30.exp5.particle.mp4 -y
rm ./tmp/freq18k20k.dist0_30.exp5.particle.avi
