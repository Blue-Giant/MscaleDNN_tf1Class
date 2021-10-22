# 保存图片的一些设置
isShowPic = 1
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88-Leftp
Heightp = 0.9-Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]


#  图片保存函数
def mySaveFig(pltm, fntmp, fp=0, ax=0, isax=0, iseps=0, isShowPic=0):
    if isax == 1:
        pltm.rc('xtick', labelsize=18)
        pltm.rc('ytick', labelsize=18)
        ax.set_position(pos, which='both')
    fnm = '%s.png'%(fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = '%s.eps'%(fntmp)
        pltm.savefig(fnm, format='eps', dpi=600)
    if fp != 0:
        fp.savefig("%s.pdf"%(fntmp), bbox_inches='tight')
    if isShowPic == 1:
        pltm.show()
    elif isShowPic == -1:
        return
    else:
        pltm.close()


#  日志记数函数
def log_string(out_str, log_out):
    log_out.write(out_str + '\n')  # 将字符串写到文件log_fileout中去，末尾加换行
    log_out.flush()                # 清空缓存区
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。
