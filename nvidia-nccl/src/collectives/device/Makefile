#
# Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
#

include ../../../makefiles/common.mk
include ../../../makefiles/version.mk

BUILDDIR ?= $(abspath ../../../build)
OBJDIR := $(BUILDDIR)/obj/collectives/device

DEPFILES   := $(patsubst %.cu, $(OBJDIR)/%.d, $(LIBSRCFILES))
DEPENDFILES:= $(DEPFILES:%.d=%.dep)
DEVOBJ     := $(OBJDIR)/devlink.o
RULESFILE  := $(OBJDIR)/Makefile.rules

NVCUFLAGS  += -I. -I.. -I$(BUILDDIR)/include -I../../include --compiler-options "-fPIC -fvisibility=hidden"


all:

# Dummy rule so that the extra dependency (%.dep) files are preserved by make
all_deps: $(DEPENDFILES)

# Auto-generating the rules per op/reduction/datatype/algorithm
$(RULESFILE) :
	@printf "Generating %-35s > %s\n" rules $@
	@mkdir -p $(OBJDIR)
#	@./gen_rules.sh $(OBJDIR) > $@

-include $(RULESFILE)

LIBOBJ     := $(GENOBJS)

-include $(DEPFILES)

# We do not want make to build *.d when running make clean.
# So we only provide targets for .dep which will produce .dep and .d,
# with only .d being included, and .dep keeping track of what needs to
# be regenerated.
$(OBJDIR)/%.dep : %.cu
	@mkdir -p $(OBJDIR)
	@$(NVCC) $(NVCUFLAGS) -M $< -o $@.tmp
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $@.tmp > $@
	@sed -e 's/.*://' -e 's/\\$$//' < $@.tmp | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $@
	@rm -f $@.tmp
	@cp $@ $(@:.dep=.d)

# Compiled kernels and collectives with relocatable device code ...
# ... and create the device-side linked object with all those.
$(DEVOBJ) : $(LIBOBJ)

clean:
	rm -f $(LIBOBJ) $(DEVOBJ) $(DEPFILES) $(DEPENDFILES) $(RULESFILE) $(STATICLIB)
