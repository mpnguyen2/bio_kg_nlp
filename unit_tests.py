import unittest
from generate_files import *

class TestIO(unittest.TestCase):

    def test_read_reltypes(self):
        # Generate and check umls_reltypes.txt
        generate_txt()
        ekg_etypes = set()
        with open(UMLS_RELTYPES_FILE, 'r') as f:
            for line in f:
                ekg_etypes.add(line.strip().split('|')[1])
        ekg_etypes = list(ekg_etypes)
        ekg_etypes.sort()
        self.assertEqual(len(ekg_etypes), 54)

if __name__ == '__main__':
    unittest.main()