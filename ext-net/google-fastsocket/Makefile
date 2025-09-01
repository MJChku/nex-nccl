CUDA_HOME?=/usr/local/cuda
INC:=-I$(CUDA_HOME)/include \
-I/home/jcm/nex-cuda-light/nccl/src/include \
-I/home/jcm/nex-cuda-light/nccl/src/include/plugin
PLUGIN_SO:=libnccl-net.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): nccl-fastsocket/*.cc
	$(CC) $(INC) -fPIC -shared -o $@ -Wl,-soname,$(PLUGIN_SO) $^
	cp libnccl-net.so ~/nex-cuda-light/nccl/build/lib/

nccl-fastsocket/*.cc:
	git clone https://github.com/google/nccl-fastsocket.git

install: $(BUILDDIR)/lib/$(PLUGIN_SO)

$(BUILDDIR)/lib/$(PLUGIN_SO): $(PLUGIN_SO)
	@printf "Grabbing %-35s > %s\n" $< $@
	mkdir -p $(BUILDDIR)/lib
	install -m 644 $< $@

clean:
	rm -f $(PLUGIN_SO)
	rm -Rf nccl-fastsocket
