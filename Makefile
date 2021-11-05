CXXFLAGS = -I include  -std=c++11 -O3 -I/home/pei_group/anaconda3/include/python3.8 -I/home/pei_group/anaconda3/include/python3.8  -Wno-unused-result -Wsign-compare -march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong  -O3 -ffunction-sections -pipe -isystem /home/pei_group/anaconda3/include -fdebug-prefix-map=/tmp/build/80754af9/python_1593706424329/work=/usr/local/src/conda/python-3.8.3 -fdebug-prefix-map=/home/pei_group/anaconda3=/usr/local/src/conda-prefix -fuse-linker-plugin -ffat-lto-objects -flto-partition=none -flto -DNDEBUG -fwrapv -O3 -Wall

LDFLAGS = -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = lanms/adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
