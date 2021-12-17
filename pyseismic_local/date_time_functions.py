import datetime


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args,
          **kwargs)


# 将运行时间换算成时分秒便于识别的形式
def time2HMS(elapsed_time=2311420.123):
    elapsed_time_h = int(elapsed_time / 3600)
    elapsed_time_m = int((elapsed_time - elapsed_time_h * 3600) / 60)
    elapsed_time_s = int(
        elapsed_time - elapsed_time_h * 3600 - elapsed_time_m * 60)
    elapsed_time_ms = int((
                                      elapsed_time - elapsed_time_h * 3600 - elapsed_time_m * 60 - elapsed_time_s) * 1e3)

    elapsed_time_str = '{:0>3d}'.format(
        elapsed_time_h) + 'h' + '{:0>2d}'.format(
        elapsed_time_m) + 'm' + '{:0>2d}'.format(
        elapsed_time_s) + 's' + '{:0>3d}'.format(elapsed_time_ms) + 'ms'
    print('   Elapsed time ' + elapsed_time_str)
    return elapsed_time_str


def data_time_str_def():
    data_time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return data_time_str

