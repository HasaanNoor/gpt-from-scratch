import unittest
import pytest
import os
import json
import xml.etree.ElementTree as ET
from parameterized import parameterized


TESTS_FILE = 'tests_file.json'
TESTS = ['raschka.gpt.MultiHeadAttention', 'raschka.gpt.FeedForward', 'raschka.gpt.LayerNorm', 'torch.nn.Dropout', 'raschka.gpt.TransformerBlock']


class TestRunPytest(unittest.TestCase):
    results = {}

    @parameterized.expand(list(zip(TESTS)))
    def test(self, test_name):
        test_name = f'test_member[{test_name}]'
        if test_name not in TestRunPytest.results:
            with open(TESTS_FILE, 'w') as f:
                json.dump(TESTS, f)
            # os.environ['FROM_SCRATCH_TESTS'] = json.dumps(TESTS)
            xml_file = 'report.xml'
            exit_code = pytest.main(['-vsra', f'--junitxml={xml_file}', 'from_scratch/from_scratch.py'])

            with open(xml_file, 'r') as fp:
                tree = ET.fromstring(fp.read())

            assert tree.tag == 'testsuites'
            testsuites = tree
            for testsuite in testsuites:
                assert testsuite.tag == 'testsuite'
                print('testsuite:', testsuite.attrib)
                for testcase in testsuite:
                    assert testcase.tag == 'testcase'
                    print('testcase:', testcase.attrib)
                    TestRunPytest.results[testcase.attrib['name']] = 'Passed'
                    for failure in testcase:
                        assert failure.tag == 'failure'
                        print('failure:', failure.attrib)
                        TestRunPytest.results[testcase.attrib['name']] = failure.attrib['message']
            os.remove(TESTS_FILE)
            if all([x == 'Passed' for x in TestRunPytest.results.values()]):
                self.assertEqual(exit_code, 0, "The pytest suite failed!")

        result = TestRunPytest.results[test_name]
        # Optionally assert that the pytest suite passed (exit code 0)
        self.assertEqual(result, 'Passed', result)


if __name__ == '__main__':
    unittest.main()