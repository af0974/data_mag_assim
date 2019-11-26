import sys
import numpy as np
import copy

class DataTable:

    def __init__(self, fileName, type='histmag_compact'):
        if type == 'histmag_compact':
            self.read_histmag_compact(fileName)
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
        mask_inten = ( self.inten > 0. )
        mask_decl = ( np.abs(self.decl) >= 0. )
        mask_inc = ( np.abs(self.inc) >= 0. )
        print(mask_inten)
        print(self.inten[~(mask_inten)])
        print(mask_decl)
        print(self.decl[~(mask_decl)])
        print(mask_inc)
        print(self.inc[~(mask_inc)])

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
        print(np.shape((self.element)))
        print(self.element)
        #sys.exit()

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
        self.observatory_id = data['observatory_id']
        self.year = data['year']
        self.lat = data['lat']
        self.lon = data['lon']
        self.element = data['element']
        self.value = data['value']
        self.sigma = data['sigma']
        self.bias = data['bias']

    def __add__(self, new):
        out = copy.deepcopy(new)
        #out.origin = np.hstack( ( self.origin, new.origin ) )
        out.year = np.hstack( ( self.year, new.year ) )
        #out.posdyear = np.hstack( ( self.posdyear, new.posdyear ) )
        #out.negdyear = np.hstack( ( self.negdyear, new.negdyear ) )
        out.lat = np.hstack( ( self.lat, new.lat ) )
        out.lon = np.hstack( ( self.lon, new.lon ) )
        out.element = np.hstack( ( self.element, new.element ) )
        out.value = np.hstack( ( self.value, new.value ) )
        out.sigma = np.hstack( ( self.sigma, new.sigma ) )
        #out.source = np.hstack( ( self.source, new.source ) )

        return out
