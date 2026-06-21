import nexus

def test_nexus_version():
    assert isinstance(nexus.__version__, str)
    assert nexus.__version__ != ""
