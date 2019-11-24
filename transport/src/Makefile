#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../common.mk

##### src files
INCEXPORTS  := nccl_net.h
LIBSRCFILES :=  transport.cc shm.cc net.cc net_socket.cc 

##### lib files
LIBNAME     := tranport.so
STATICLIBNAME := libtransport_static.a
BUILDDIR ?= $(abspath ../)
INCDIR := $(BUILDDIR)/include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj

##### target files
STATICLIBTARGET := $(STATICLIBNAME)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)
LDFLAGS    += -lpthread -lrt -ldl

##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) 
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) 
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(eval TMP := $(shell mktemp -d))
	cp $(LIBOBJ) $(TMP)
	ar cr $@ $(LIBOBJ) $(TMP)/*.o
	rm -Rf $(TMP)

$(OBJDIR)/%.o : %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf ${INCDIR} ${LIBDIR} ${OBJDIR}

install : lib
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/
