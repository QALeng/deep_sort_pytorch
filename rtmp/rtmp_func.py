import subprocess as sp

#管道初始化
def pipe_init(rtmp_url,size_str,fps,bit):
    command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', size_str,
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-rtsp_transport','tcp',
        '-f', 'flv',
        '-b:v',bit,
        rtmp_url]
    pipe = sp.Popen(command, stdin=sp.PIPE)
    return  pipe
