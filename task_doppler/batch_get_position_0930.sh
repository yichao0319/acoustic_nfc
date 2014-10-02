#!/bin/bash


/usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp1', 17999, 19999, [0, 1], [-0.1, 0.5], [0, 1.1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist80_0.exp1.2d_trace.avi ./tmp/freq18k20k.dist80_0.exp1.2d_trace.mp4 -y
rm ./tmp/freq18k20k.dist80_0.exp1.2d_trace.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp1', 17999, 19999, [0, 0.5], [-0.1, 0.5], [0, 1]); exit;"
ffmpeg -i ./tmp/freq18k20k.dist0_30.exp1.2d_trace.avi ./tmp/freq18k20k.dist0_30.exp1.2d_trace.mp4 -y
rm ./tmp/freq18k20k.dist0_30.exp1.2d_trace.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.pic1', 17998.9, 19999, [0, 0.5], [-0.1, 0.2], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic1.2d_trace.avi ./tmp/freq18k20k.pic1.2d_trace.mp4 -y
rm ./tmp/freq18k20k.pic1.2d_trace.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.pic2', 17999, 19999, [0, 0.5], [-0.1, 0.1], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic2.2d_trace.avi ./tmp/freq18k20k.pic2.2d_trace.mp4 -y
rm ./tmp/freq18k20k.pic2.2d_trace.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.pic3', 17999, 19999, [0, 0.5], [-0.05, 0.3], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic3.2d_trace.avi ./tmp/freq18k20k.pic3.2d_trace.mp4 -y
rm ./tmp/freq18k20k.pic3.2d_trace.avi

/usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.pic4', 17999, 19999, [0, 0.5], [-0.1, 0.2], [0.3, 0.6]); exit;"
ffmpeg -i ./tmp/freq18k20k.pic4.2d_trace.avi ./tmp/freq18k20k.pic4.2d_trace.mp4 -y
rm ./tmp/freq18k20k.pic4.2d_trace.avi


###################################################

# get_position('../data/rcv_pkts/exp0930/', 'freq18k17k.static.exp1', 17999, 16999, [0, 0.5], [-0.1, 0.5], [0, 1.1]);

# get_position('../data/rcv_pkts/exp0930/', 'freq18k17k.line.exp1', 17999, 16999, [0, 1], [-0.1, 0.5], [0, 1.1]);

# get_position('../data/rcv_pkts/exp0930/', 'freq18k17k.square.exp1', 17999, 16999, [0, 0.5], [-0.1, 0.5], [0, 1.1]);


# get_position('../data/rcv_pkts/exp0930/', 'freq18k20k_17k19k.static.exp1', [17999, 19999], [16999, 18999], [0, 0.5], [-0.1, 0.5], [0, 1.1]);

# get_position('../data/rcv_pkts/exp0930/', 'freq18k20k_17k19k.line.exp1', [17999, 19999], [16999, 18999], [0, 1], [-0.1, 0.5], [0, 1.1]);

# get_position('../data/rcv_pkts/exp0930/', 'freq18k20k_17k19k.square.exp1', [17999, 19999], [16999, 18999], [0, 0.5], [-0.1, 0.5], [0, 1.1]);



# get_position('../data/rcv_pkts/exp0930/', 'freq18k20k_17k19k.rotate.exp1', [17999], [16999], [0, 0.5], [-0.1, 0.5], [0, 1.1]);

# get_position('../data/rcv_pkts/exp0930/', 'freq18k20k_17k19k.circle.exp1', [17999, 19999], [16999, 18999], [0, 0.7], [-0.3, 0.3], [0.2, 0.8]);

# get_position('../data/rcv_pkts/exp0930/', 'freq18k20k_17k19k.line.exp3', [17999], [16999], [0, 1], [-0.1, 0.3], [0, 1.1]);


###################################################

# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp2', 17999, 19999, [0, 1], [-0.1, 0.5], [0, 1.1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist80_0.exp2.2d_trace.avi ./tmp/freq18k20k.dist80_0.exp2.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist80_0.exp2.2d_trace.avi

# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp2', 17999, 19999, [0, 0.5], [-0.1, 0.5], [0, 1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist0_30.exp2.2d_trace.avi ./tmp/freq18k20k.dist0_30.exp2.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist0_30.exp2.2d_trace.avi


# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp3', 17999, 19999, [0, 1], [-0.1, 0.5], [0, 1.1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist80_0.exp3.2d_trace.avi ./tmp/freq18k20k.dist80_0.exp3.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist80_0.exp3.2d_trace.avi

# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp3', 17999, 19999, [0, 0.5], [-0.1, 0.5], [0, 1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist0_30.exp3.2d_trace.avi ./tmp/freq18k20k.dist0_30.exp3.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist0_30.exp3.2d_trace.avi


# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp4', 17999, 19999, [0, 1], [-0.1, 0.5], [0, 1.1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist80_0.exp4.2d_trace.avi ./tmp/freq18k20k.dist80_0.exp4.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist80_0.exp4.2d_trace.avi

# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp4', 17999, 19999, [0, 0.5], [-0.1, 0.5], [0, 1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist0_30.exp4.2d_trace.avi ./tmp/freq18k20k.dist0_30.exp4.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist0_30.exp4.2d_trace.avi


# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist80_0.exp5', 17999, 19999, [0, 1], [-0.1, 0.5], [0, 1.1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist80_0.exp5.2d_trace.avi ./tmp/freq18k20k.dist80_0.exp5.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist80_0.exp5.2d_trace.avi

# /usr/local/MATLAB/R2013b/bin/matlab -r "get_position('../data/rcv_pkts/exp0930/', 'freq18k20k.dist0_30.exp5', 17999, 19999, [0, 0.5], [-0.1, 0.5], [0, 1]); exit;"
# ffmpeg -i ./tmp/freq18k20k.dist0_30.exp5.2d_trace.avi ./tmp/freq18k20k.dist0_30.exp5.2d_trace.mp4 -y
# rm ./tmp/freq18k20k.dist0_30.exp5.2d_trace.avi
