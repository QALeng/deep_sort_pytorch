import subprocess as sp



def pipeFuncInit(rtmpUrl,sizeStr,fps):
    command = ['ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', sizeStr,
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'ultrafast',
        '-f', 'flv',
        rtmpUrl]
    #管道特性配置
    # pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)
    pipe = sp.Popen(command, stdin=sp.PIPE) #,shell=False
    # pipe.stdin.write(frame.tostring())
    # 结果帧处理 存入文件 / 推流 / ffmpeg 再处理
    return  pipe
