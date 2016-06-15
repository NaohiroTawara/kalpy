from pynvml import *


def SelectGpuIdAuto():
    '''
    Returns list of GPU IDs that currently have no process. 
    List is sorted in a descending the memory.
    '''
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    freeDevices={}
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        memInfo = nvmlDeviceGetMemoryInfo(handle)
        ratio = float(memInfo.free) / float(memInfo.total)
        print "Device" + str(i) + ": " + nvmlDeviceGetName(handle),  \
            '\tfree:' + str(memInfo.total/1000000) + \
            'M, used:'  + str(memInfo.used/1000000)  + \
            'M, total:' + str(memInfo.total/1000000) + \
            'M, free/total: %0.6f' % ratio
        try:
            if not len(nvmlDeviceGetComputeRunningProcesses(handle))>0:
                freeDevices[i] = ratio
        except:
            print 'Not supported device: %s' % nvmlDeviceGetName(handle)
    sorted_devices = [k for k,v in sorted(freeDevices.items(), key=lambda x:x[1], reverse=True)]
    print 'Trying to select device: '   + str(sorted_devices[0]) + \
        ' (automatically), mem_ratio: %0.6f' % freeDevices[sorted_devices[0]]
    return sorted_devices

if __name__ == "__main__":
    SelectGpuIdAuto()
