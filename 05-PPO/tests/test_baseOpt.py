import unittest
import sys,os
import numpy as np
dir=os.path.abspath(os.path.dirname(__file__) + './..')
sys.path.append(dir)


class TestBaseOpt(unittest.TestCase):
    def setUp(self):
        pass

        
    def test_numpy_max(self): 
        data=np.arange(24).reshape(4,3,2)
        '''
        [...,0]:
         0  2  4
         6  8 10
        12 14 16
        18 20 22
        [...,1]:
         1  3  5
         7  9 11
        13 15 17
        19 21 23
        '''
        #print(data[...,0])
        self.assertEqual(data.max(),23)
        md=data.max(axis=0)
        self.assertEqual(md.shape,(3,2))
        #print(md[0,:])
        self.assertEqual(list(md[0,:]),[18,19])
        self.assertEqual(list(md[:,0]),[18,20,22])
        self.assertEqual(list(md[:,1]),[19,21,23])
        self.assertEqual(list(md[...,0]),[18,20,22])
        #print(data.max(axis=1))
        md=data.max(axis=1)
        self.assertEqual(md.shape,(4,2))
        self.assertEqual(list(md[:,0]),[4,10,16,22])
        self.assertEqual(list(md[:,1]),[5,11,17,23])
        #print(data.max(axis=2))
        md=data.max(axis=2)
        self.assertEqual(md.shape,(4,3))
        self.assertEqual(list(md[:,0]),[1,7,13,19])
        self.assertEqual(list(md[0,:]),[1,3,5])

        




        
if __name__ == '__main__':
    unittest.main()