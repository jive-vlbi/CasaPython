import tasks
import clog_classified


anmsname = 'n14p1.ms'
tasks.applycal(anmsname, gaintable=['n14p1.gcal', 'n14p1.tsys'], parang=True)
ff = clog_classified.FringeFitter(anmsname, 'numeric-sbd.fj', ref_antenna_name='WB', scans=[1], snr_threshold=10.0, solint=240)
ff.run()

tasks.applycal(anmsname, gaintable=['n14p1.gcal', 'n14p1.tsys'], parang=True)

import applyfringe

for anffd in ff.all_ffds:
    app = applyfringe.Applier(anffd.msname, anffd.antenna_list,
                              anffd.swid, anffd.pol_id, anffd.polind,
                              anffd.qrest, anffd.get_ref_time(),
                              anffd.phs, anffd.dels, anffd.rs, anffd.disps,
                              'CORRECTED_DATA')
    app.applyCalsToMS()

for anffd in ff.all_ffds:
    print anffd.phs, anffd.dels, anffd.rs, anffd.disps
