from bliss.setup_globals import *
from os.path import isdir
from os import mkdir
import glob

##helpers functions


def trigger_opiom2_tpg():
      opiom2_tpg.opiom.comm_ack("CNT 2 RESET")
      opiom2_tpg.opiom.comm_ack("CNT 2 CLK8 RISE D32 PULSE 1000 1000 1")
      opiom2_tpg.opiom.comm_ack("CNT 2 START")

def trigger_opiom2_tpglong():
      opiom2_tpg.opiom.comm_ack("CNT 2 RESET")
      opiom2_tpg.opiom.comm_ack("CNT 2 CLK8 FALL D32 PULSE 450000 450000 1")
      opiom2_tpg.opiom.comm_ack("CNT 2 START")


def shpb_arm_scopes():

      scopeid19.device.send("ARM")
      scopePool.device.send("ARM")
      sleep(1)

def shimadzuLargeFoV():
      shclose()
      umv(yc, 200) #HRtomo
      wbtf.setout() #ID19
      umv(pshg, 3, psvg, 2.5) #wherever
      umv(sshg, 11.66, ssvg, 8.05) #wherever
      umv(z3rd, 4.5) #ID19
            
def shimadzuHighRes():
      shclose()
      umv(yc, 0) #HRtomo
      umv(pshg, 2, psvg, 2) #wherever
      umv(sshg, 1.66, ssvg, 2.55) #wherever
      umv(z3rd, 7.8) #ID19
      wbtf.setin(2) #ID19
      wbtf.setin(1) #ID19

def PhotronTomo():
      shclose()
      umv(cz, 81.796100) 
      umv(sshg, 10.42563, ssvg, 9.22563)
      umv(u13a_gap, 14.3)
      umv(yrot, 0) #no half tomo
      update_off()
      isg_shutter.fsextoff()
      print(' use isg_shutter.open() and isg_shutter.close() ')

def TomoTomo():
      shclose()
      umv(cz, -3.235400) 
      umv(sshg, 9.42563, ssvg, 5.22563)
      umv(u13a_gap, 21.0)
      update_on()
      isg_shutter.close()
      isg_shutter.fsexton()
      umv(yrot, 3) #half tomo
      print(' shopen() now required ! ')
      
def move_to_bi10():
      shclose()
      #zc1_rev = 30.714999999999996
      yc_rev = 318.6
      sshg_rev = 3.8
      ssvg_rev = 2.3
      pshg_rev = 0.95
      psvg_rev = 0.75
      #umv(u13a_gap, 14)
      umv(yc, yc_rev, sshg, sshg_rev, ssvg, ssvg_rev, pshg, pshg_rev, psvg, psvg_rev)
      print(' shopen() now required ! ')
      
def move_to_photron():
      shclose()
      zc2_ph = 1.30
      yc_ph = -15.69
      sshg_ph = 2.9
      ssvg_ph = 2.9
      pshg_ph = 1.95
      psvg_ph = 1.75
      
      umv(zc2, zc2_ph, yc, yc_ph, sshg, sshg_ph, ssvg, ssvg_ph, xc, 300, pshg, pshg_ph, psvg, psvg_ph)
      umv(u13a_gap, 11.8)
      print(' shopen() now required ! ')
      
def move_to_dimax():
      shclose()
      yc_ph = -15.69
      sshg_ph = 5.35
      ssvg_ph = 5.45
      pshg_ph = 1.125
      psvg_ph = 1.2
      umv(yc, yc_ph, sshg, sshg_ph, ssvg, ssvg_ph, pshg, pshg_ph, psvg, psvg_ph)
      print(' shopen() now required ! ')
      


def move_to_10X():
   # shclose()
    sshg_10 = 2.2
    ssvg_10 = 1.2
    pshg_10 = 0.475
    psvg_10 = 0.45
    mrtriplemic.objective = 2
    umv(sshg, sshg_10, ssvg, ssvg_10, pshg, pshg_10, psvg, psvg_10)
    
def move_to_20X():
    #shclose()
    sshg_20 = 1.1
    ssvg_20 = 0.95
    pshg_20 = 0.475
    psvg_20 = 0.45
    mrtriplemic.objective = 1
    umv(sshg, sshg_20, ssvg, ssvg_20, pshg, pshg_20, psvg, psvg_20)

shutter_in_position=0.0 #4.5 1x Shimadzu, 7.8 5x Shimadzu

def series_of_tomo(name, n_scans, dz = 0):
    newcollection(name)
    full_tomo.pars.dark_at_start = True
    full_tomo.pars.flat_at_start = True
    try:
        i = 0
        while n_scans == -1 or i < n_scans:
            if n_scans == -1:
                print("scan ",str(i),"/ unlimited")
            else:
                print("scan ",str(i),"/",str(n_scans))
            newdataset("first_position")
            full_tomo.run()
            full_tomo.pars.dark_at_start = False
            full_tomo.pars.flat_at_start = False
            if not dz == 0:
                newdataset("second_position")
                umvr(sz, dz)
                full_tomo.run()
                umvr(sz, -dz)
            i += 1
    finally:
        full_tomo.pars.dark_at_start = True
        full_tomo.pars.flat_at_start = True

def laser_IN_shutter_OUT():
      umv(laser,0,z3rd,100)

def laser_OUT_shutter_IN():
      umv(laser,-130,z3rd,shutter_in_position)

def laser_IN():
      umv(laser,0)

def laser_OUT():
      umv(laser,-130)


## main macros

def shutter_for_flat(atime):
      if z3rd.position > shutter_in_position:
            print('##########################################################################')
            print('\n #####   Shutter not in place!!! Check laser and shutter    !!!      ##### \n')
            print('##########################################################################')
      elif frontend.is_closed:
            print('##########################################################################')
            print('\n #####   FRONTEND is closed  ! Check the machine status  ##### \n')
            print('##########################################################################')
      elif bsh1.is_closed:
            print('##########################################################################')
            print('\n #####   First Safety shutter is closed  ! Check status-->Open manually ( bsh1.open() )##### \n')
            print('##########################################################################')
      else:
            isg_shutter.fsextoff()
            sleep(1)
            isg_shutter.open()
            sleep(atime)
            isg_shutter.close()
            sleep(1)
            isg_shutter.fsexton()



def pp_shot(shot_name):

      print('####################\n')
      if z3rd.position > shutter_in_position:
            print('##########################################################################')
            print('\n #####   Shutter not in place!!! Check laser and shutter    !!!      ##### \n')
            print('##########################################################################')
            umv(z3rd,shutter_in_position)
      elif laser.position == 0:
            print('##########################################################################')
            print('\n #####   Laser still in place!!! Check laser and shutter    !!!      ##### \n')
            print('##########################################################################')
            umv(laser,-130)
      elif frontend.is_closed:
            print('##########################################################################')
            print('\n #####   FRONTEND is closed  ! Check the machine status  ##### \n')
            print('##########################################################################')
      elif bsh1.is_closed:
            print('##########################################################################')
            print('\n #####   First Safety shutter is closed  ! Check status-->Open manually ( bsh1.open() )##### \n')
            print('##########################################################################')
      elif isg_shutter.state_string == 'Open':
            print('##########################################################################')
            print('\n #####   Fast shutter is Open  ! Check status-->Close manually ( isg_shutter.state/isg_shutter.mode )##### \n')
            print('##########################################################################')
            bsh1.close()
      else:

        
            #open safety shutter 
        
            trigger_opiom2_tpg()

            #close safety shutter
            bsh2.close()

            sleep(0.2)
            newcollection(shot_name)
            sct(0.3,bunch_currents_controller)



pr = {}

pr['LR'] = {}
pr['HR'] = {}

pr['LR']['px_size'] = 3.104
pr['HR']['px_size'] = 0.642

pr['LR']['yc_pos'] = 32.53
pr['HR']['yc_pos'] = 332.53

pr['LR']['xc_pos'] = 450
pr['HR']['xc_pos'] = 100


pr['LR']['ss_vg'] = 6.375
pr['LR']['ss_hg'] = 7.45

pr['HR']['ss_vg'] = 1.575
pr['HR']['ss_hg'] = 1.35


pr['LR']['u17_6c_gap'] = 35.5
# Gap
pr['HR']['u17_6c_gap'] = 29


pr['LR']['par_tomo'] = {}
pr['HR']['par_tomo'] = {}


pr['LR']['par_tomo']['tomo_n'] = 6000
# Number of projections
pr['HR']['par_tomo']['tomo_n'] = 6000

pr['LR']['par_tomo']['exposure_time'] = 0.1
# Exposure time
pr['HR']['par_tomo']['exposure_time'] = 0.1

pr['LR']['sample_detector_distance'] = 2600
# Distance
pr['HR']['sample_detector_distance'] = 100

pr['LR']['par_tomo']['latency_time'] = 0.006
pr['HR']['par_tomo']['latency_time'] = 0.006

pr['LR']['par_tomo']['energy'] = 19
pr['HR']['par_tomo']['energy'] = 19

pr['LR']['halftomo_pos'] = ((2560/2.0)*0.8)*(pr['LR']['px_size']/1000)
pr['HR']['halftomo_pos'] = ((2048/2.0)*0.8)*(pr['HR']['px_size']/1000)


def moveSamplePos(res,dx,dy,dz,x0,y0,z0,image_size = [2560,2560,2160],sign_x=-1,sign_y=-1,sign_z=1):

    px_size = pr[res]['px_size']
    x_trans = sign_x*((dx - (image_size[0]/2.0))*px_size/1000.0)
    y_trans = sign_y*((dy - (image_size[1]/2.0))*px_size/1000.0)
    z_trans = sign_z*((dz - (image_size[2]/2.0))*px_size/1000.0)
    umv(sx,x0,sy, y0,sz,z0)
    umvr(sx,y_trans, sy, x_trans,sz,z_trans)


def move2LR(flag_halftomo=False):#, shift):

    tomoccdselect(pcolid19det3)
    DISTANCE(pr['LR']['sample_detector_distance'])

    dic_mrtomo = mrfull_tomo.pars.to_dict()

    for key in pr['LR']['par_tomo'].keys():
        dic_mrtomo[key] = pr['LR']['par_tomo'][key]

    mrfull_tomo.pars.from_dict(dic_mrtomo)

    if flag_halftomo:
        mrfull_tomo.pars.half_acquisition = True
        mrfull_tomo.pars.shift_in_mm = pr['LR']['halftomo_pos']
        umv(yrot, pr['LR']['halftomo_pos'])
    else:
        mrfull_tomo.pars.half_acquisition = False
        umv(yrot,0)

    umv(yc,pr['LR']['yc_pos'],ssvg, pr['LR']['ss_vg'],sshg, pr['LR']['ss_hg'],xc, pr['LR']['xc_pos'])
    umv(u17_6c_gap,pr['LR']['u17_6c_gap'])
    umv(zdeco, -20)


def move2HR(flag_halftomo=True):#, shift):


    tomoccdselect(pco42win)
    DISTANCE(pr['HR']['sample_detector_distance'])
    
    dic_mrtomo = mrfull_tomo.pars.to_dict()

    for key in pr['HR']['par_tomo'].keys():
        dic_mrtomo[key] = pr['HR']['par_tomo'][key]

    mrfull_tomo.pars.from_dict(dic_mrtomo)

    if flag_halftomo:
        mrfull_tomo.pars.half_acquisition = True
        mrfull_tomo.pars.shift_in_mm = pr['HR']['halftomo_pos']
        mrfull_tomo.pars.acquisition_position = pr['HR']['halftomo_pos']
        umv(yrot, pr['HR']['halftomo_pos'])
    else:
        mrfull_tomo.pars.half_acquisition = False
        mrfull_tomo.pars.acquisition_position = 0.0
        umv(yrot,0)

    umv(yc,pr['HR']['yc_pos'],ssvg, pr['HR']['ss_vg'],sshg, pr['HR']['ss_hg'],xc, pr['HR']['xc_pos'])
    umv(u17_6c_gap,pr['HR']['u17_6c_gap'])
    umv(zdeco, 0)
    

def do_multiple_HR_scans(res,source_scan,scan_positions,x0,y0,z0,image_size = [2560,2560,2160],sign_x=-1,sign_y=-1,sign_z=1):
    
    move2HR(flag_halftomo=True)
    for ii, scan_position in enumerate(scan_positions):
        dx, dy, dz = scan_position
        moveSamplePos(res,dx,dy,dz,x0,y0,z0,image_size,sign_x,sign_y,sign_z)
        print("%s_HR_ROI_%i_%i_%i" %(source_scan,dx,dy,dz))
        newcollection(source_scan)
        newdataset("%s_HR_ROI_%i_%i_%i" %(source_scan,dx,dy,dz))
        full_tomo.full_turn_scan()
    umv(sx, x0, sy, y0, sz, z0)


samples = { 
            "C3":"P714_virg",
            "C4":"P714_lact",
            }
zStep=5
nScans=1

def launch_sample_changer2( ):

    for sp,sn in samples.items():
        print(sp, sn)
        if sn is not None:
            samplechanger.load_sample(sp)
            newcollection(f"{sn}")
            ## ~ zseries(zStep,nScans)
            move2HR()
            fulltomo360()
            
            samplechanger.unload_sample()

def tester2():
    move2HR()

def slits():
    wm(pshg, psvg, psho, psvo)
    wm(sshg, ssvg, ssho, ssvo)
      
      
def srot_modulo360():
    srot.position = srot.position%360
