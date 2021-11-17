import sys
import numpy as np
import copy

class DataTable:

    def __init__(self, fileName, type='histmag_compact'):
        if type == 'histmag_compact':
            self.read_histmag_compact(fileName)
        if type == 'histmag_complete':
            self.read_histmag_complete(fileName)
        if type == 'obsmag_gufm1':
            self.read_obsdata_gufm1(fileName)


    def read_histmag_compact(self, fileName):
        datatype = ([
        ('id', int),
        ('origin',int),
        ('year', float),
        ('posdyear', float),
        ('negdyear', float),
        ('lat', float),
        ('lon', float),
        ('decl', float),
        ('ddecl', float),
        ('inc', float),
        ('dinc', float),
        ('inten', float),
        ('dinten', float),
        ('inten_code','S4'),
        ('source','S18'),
        ('ref1','S250'),
        ('user_comment','S10')
        ])
        data = np.genfromtxt(fileName, skip_header=1, delimiter='\t', dtype=datatype )
        self.id = data['id']
        self.origin = data['origin']
        self.year = data['year']
        self.posdyear = data['posdyear']
        self.negdyear = data['negdyear']
        self.lat = data['lat']
        self.lon = data['lon']
        self.decl = data['decl']
        self.ddecl = data['ddecl']
        self.inc = data['inc']
        self.dinc = data['dinc']
        self.inten = data['inten']
        self.dinten = data['dinten']
        self.inten_code = np.char.decode(data['inten_code'])
        self.source = np.char.decode(data['source'])
        mask_inten = ~np.isnan((self.inten))
        mask_decl = ~np.isnan((self.decl))
        mask_inc = ~np.isnan((self.inc))
        # unfold arrays to make them gufm1-like
        titi = np.array(data['origin'][mask_inten])
        toto = np.array(data['origin'][mask_decl])
        tutu = np.array(data['origin'][mask_inc])
        self.origin = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['year'][mask_inten])
        toto = np.array(data['year'][mask_decl])
        tutu = np.array(data['year'][mask_inc])
        self.year = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['posdyear'][mask_inten])
        toto = np.array(data['posdyear'][mask_decl])
        tutu = np.array(data['posdyear'][mask_inc])
        self.posdyear = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['negdyear'][mask_inten])
        toto = np.array(data['negdyear'][mask_decl])
        tutu = np.array(data['negdyear'][mask_inc])
        self.negdyear = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['lat'][mask_inten])
        toto = np.array(data['lat'][mask_decl])
        tutu = np.array(data['lat'][mask_inc])
        self.lat = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['lon'][mask_inten])
        toto = np.array(data['lon'][mask_decl])
        tutu = np.array(data['lon'][mask_inc])
        self.lon = np.hstack( ( titi, toto, tutu) ) 
        titi = np.ones_like(data['lon'][mask_inten], dtype=str)
        titi[:] = 'F'
        toto = np.ones_like(data['lon'][mask_decl], dtype=str)
        toto[:] = 'D'
        tutu = np.ones_like(data['lon'][mask_inc], dtype=str)
        tutu[:] = 'I'
        self.element = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['inten'][mask_inten])
        toto = np.array(data['decl'][mask_decl])
        tutu = np.array(data['inc'][mask_inc])
        self.value = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['dinten'][mask_inten])
        toto = np.array(data['ddecl'][mask_decl])
        tutu = np.array(data['dinc'][mask_inc])
        self.sigma = np.hstack( ( titi, toto, tutu) )  
        self.month = np.zeros_like(self.year)
        self.day = np.zeros_like(self.year)

    def read_histmag_complete(self,fileName):
        datatype = ([
        ('id', int),
        ('origin', float),
        ('year', float),
        ('posdyear', float),
        ('negdyear', float),
        ('dating_meth', 'S100'), 
        ('dyearCalc', 'S18'),
        ('month', float),
	('day', float),
        ('hour', 'S8'),
        ('seq', 'S10'),
        ('prev', 'S10'),
        ('next', 'S10'),
        ('equal', 'S10'),
        ('lat', float),
        ('lon', float),
        ('site', 'S100'), 
        ('location', 'S60'), 
        ('country', 'S30'), 
        ('orig_lon', float),
        ('merid', 'S10'), 
        ('lon_ed', 'S1'),
        ('decl', float),
        ('ddecl', float),
        ('decl_meth', 'S30'),
        ('no_decl', int),
        ('decl_inst', 'S60'),
        ('inc', float),
        ('dinc', float),
        ('inc_meth', 'S20'),
        ('no_inc', int),
        ('inc_inst', 'S30'),
        ('k', float), 
        ('a95', float),
        ('nosp_dir_meas', int),
        ('nosp_dir_acc', int),
        ('dir_analysis', 'S100'), 
        ('inten', float),
        ('dinten', float),
        ('inten_code','S4'),
        ('cal', float),
        ('inten_meth', 'S100'), 
        ('no_inten', int),
        ('inten_inst', 'S100'),
        ('nosp_inten_meas', int),
        ('nosp_inten_acc', int),
        ('alteration', 'S100'), 
        ('coolingrate', 'S100'),
        ('anisotropy', 'S100'),
        ('MD_monitor', 'S100'),
        ('source','S21'),
        ('obs', 'S30'),
        ('comment', 'S100'), #53
        ('pubID', 'S100'), #54
        ('comp_id', 'S20'), #55
        ('flag', int), #56
        ('flag_comm', 'S20'), #57
        ('user_comment','S100') #58
        ])
        data = np.genfromtxt(fileName, skip_header=1, delimiter='\t', dtype=datatype, comments=None, skip_footer=15)
        self.id = data['id']
        self.origin = data['origin']
        self.year = data['year']
        self.month = data['month']
        self.day = data['day']
        #self.year = dt(data['year'], data['month'], data['day'])
        toto = (~np.isnan(self.month)) & (~np.isnan(self.day)) & (self.day!=0.) 
        #print(self.year[toto])
        #print(self.month[toto])
        #print(self.day[toto])
        #self.year[toto] = dt(np.array(self.year[toto]), np.array(self.month[toto]), np.array(self.day[toto]))
        #print(self.year[toto])
        # not taking the hour into account yet, but when we do it, consider day-1       
        self.posdyear = data['posdyear']
        self.negdyear = data['negdyear']
        self.dyearCalc = data['dyearCalc']
        self.lat = data['lat']
        self.lon = data['lon']
        self.decl = data['decl']
        self.ddecl = data['ddecl']
        self.inc = data['inc']
        self.dinc = data['dinc']
        self.inten = data['inten']
        self.dinten = data['dinten']
        self.inten_code = np.char.decode(data['inten_code'])
        self.source = np.char.decode(data['source'])
        mask_inten = ~np.isnan((self.inten))
        mask_decl = ~np.isnan((self.decl))
        mask_inc = ~np.isnan((self.inc))
        # unfold arrays to make them gufm1-like
        titi = np.array(data['origin'][mask_inten])
        toto = np.array(data['origin'][mask_decl])
        tutu = np.array(data['origin'][mask_inc])
        self.origin = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['year'][mask_inten])
        toto = np.array(data['year'][mask_decl])
        tutu = np.array(data['year'][mask_inc])
        self.year = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['month'][mask_inten])
        toto = np.array(data['month'][mask_decl])
        tutu = np.array(data['month'][mask_inc])
        self.month = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['day'][mask_inten])
        toto = np.array(data['day'][mask_decl])
        tutu = np.array(data['day'][mask_inc])
        self.day = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['posdyear'][mask_inten])
        toto = np.array(data['posdyear'][mask_decl])
        tutu = np.array(data['posdyear'][mask_inc])
        self.posdyear = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['negdyear'][mask_inten])
        toto = np.array(data['negdyear'][mask_decl])
        tutu = np.array(data['negdyear'][mask_inc])
        self.negdyear = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['dyearCalc'][mask_inten])
        toto = np.array(data['dyearCalc'][mask_decl])
        tutu = np.array(data['dyearCalc'][mask_inc])
        self.dyearCalc = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['lat'][mask_inten])
        toto = np.array(data['lat'][mask_decl])
        tutu = np.array(data['lat'][mask_inc])
        self.lat = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['lon'][mask_inten])
        toto = np.array(data['lon'][mask_decl])
        tutu = np.array(data['lon'][mask_inc])
        self.lon = np.hstack( ( titi, toto, tutu) )
        titi = np.ones_like(data['lon'][mask_inten], dtype=str)
        titi[:] = 'F'
        toto = np.ones_like(data['lon'][mask_decl], dtype=str)
        toto[:] = 'D'
        tutu = np.ones_like(data['lon'][mask_inc], dtype=str)
        tutu[:] = 'I'
        self.element = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['inten'][mask_inten])
        toto = np.array(data['decl'][mask_decl])
        tutu = np.array(data['inc'][mask_inc])
        self.value = np.hstack( ( titi, toto, tutu) )
        titi = np.array(data['dinten'][mask_inten])
        toto = np.array(data['ddecl'][mask_decl])
        tutu = np.array(data['dinc'][mask_inc])
        self.sigma = np.hstack( ( titi, toto, tutu) ) 
        titi = np.array(data['source'][mask_inten])
        toto = np.array(data['source'][mask_decl])
        tutu = np.array(data['source'][mask_inc])
        self.source = np.hstack( ( titi, toto, tutu) )

    def read_obsdata_gufm1(self,fileName):
        datatype = ([
        ('observatory_id','S4'),
        ('year', float),
        ('lat', float),
        ('lon', float),
        ('element', 'S1'),
        ('value', float),
        ('sigma', float),
        ('bias', float)
        ])
        data = np.genfromtxt(fileName, dtype=datatype )
        self.source = data['observatory_id']
        self.year = data['year']
        self.lat = data['lat']
        self.lon = data['lon']
        self.element = data['element']
        self.value = data['value']
        self.sigma = data['sigma']
        self.bias = data['bias']
        toto = np.ones_like(data['lon'], dtype=int)
        toto[:] = 2
        self.origin = toto
        self.posdyear = np.zeros_like(data['year'])
        self.negdyear = np.zeros_like(data['year'])
        self.month = np.zeros_like(data['year'])
        self.day = np.zeros_like(data['year'])
        self.dyearCalc = np.zeros_like(data['year'])

    def __add__(self, new):
        out = copy.deepcopy(new)
        out.origin = np.hstack( ( self.origin, new.origin ) )
        out.year = np.hstack( ( self.year, new.year ) )
        out.month = np.hstack( ( self.month, new.month ) )
        out.day = np.hstack( ( self.day, new.day ) )
        out.posdyear = np.hstack( ( self.posdyear, new.posdyear ) )
        out.negdyear = np.hstack( ( self.negdyear, new.negdyear ) )
        out.dyearCalc = np.hstack( ( self.dyearCalc, new.dyearCalc ) )
        out.lat = np.hstack( ( self.lat, new.lat ) )
        out.lon = np.hstack( ( self.lon, new.lon ) )
        out.element = np.hstack( ( self.element, new.element ) )
        out.value = np.hstack( ( self.value, new.value ) )
        out.sigma = np.hstack( ( self.sigma, new.sigma ) )
        out.source = np.hstack( ( self.source, new.source ) )

        return out

    def separate_origin(self):
        out0 = copy.deepcopy(self)
        out1 = copy.deepcopy(self)
        out0.origin = self.origin[self.origin == 0]
        out0.year = self.year[self.origin == 0]
        out0.month = self.month[self.origin == 0]
        out0.day = self.day[self.origin == 0]
        out0.posdyear = self.posdyear[self.origin == 0]
        out0.negdyear = self.negdyear[self.origin == 0]
        out0.dyearCalc = self.dyearCalc[self.origin == 0]
        out0.lat = self.lat[self.origin == 0]
        out0.lon = self.lon[self.origin == 0]
        out0.element = self.element[self.origin == 0]
        out0.value = self.value[self.origin == 0]
        out0.sigma = self.sigma[self.origin == 0]
        out0.source = self.source[self.origin == 0]
        out1.origin = self.origin[self.origin == 1]
        out1.year = self.year[self.origin == 1]
        out1.month = self.month[self.origin == 1]
        out1.day = self.day[self.origin == 1]
        out1.posdyear = self.posdyear[self.origin == 1]
        out1.negdyear = self.negdyear[self.origin == 1]
        out1.dyearCalc = self.dyearCalc[self.origin == 1]
        out1.lat = self.lat[self.origin == 1]
        out1.lon = self.lon[self.origin == 1]
        out1.element = self.element[self.origin == 1]
        out1.value = self.value[self.origin == 1]
        out1.sigma = self.sigma[self.origin == 1]
        out1.source = self.source[self.origin == 1]

        return out0, out1

    def select_indx(self, ind):
        out = copy.deepcopy(self)
        out.origin = self.origin[ind]
        out.year = self.year[ind]
        out.month = self.month[ind]
        out.day = self.day[ind]
        out.posdyear = self.posdyear[ind]
        out.negdyear = self.negdyear[ind]
        out.dyearCalc = self.dyearCalc[ind]
        out.lat = self.lat[ind]
        out.lon = self.lon[ind]
        out.element = self.element[ind]
        out.value = self.value[ind]
        out.sigma = self.sigma[ind]
        out.source = self.source[ind]

        return out

    def sort_year(self):
        out = copy.deepcopy(self)
        toto = np.argsort(self.year)
        out.origin = self.origin[toto]
        out.year = self.year[toto]
        out.month = self.month[toto]
        out.day = self.day[toto]
        out.posdyear = self.posdyear[toto]
        out.negdyear = self.negdyear[toto]
        out.dyearCalc = self.dyearCalc[toto]
        out.lat = self.lat[toto]
        out.lon = self.lon[toto]
        out.element = self.element[toto]
        out.value = self.value[toto]
        out.sigma = self.sigma[toto]
        out.source = self.source[toto]

        return out

