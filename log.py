# coding:utf8
import logging

logging.getLogger("mibao").handlers = []
log = logging.getLogger('mibao')
log.setLevel(logging.DEBUG)
log.propagate = False

fmt = logging.Formatter('[%(levelname)s] %(filename)s %(lineno)s: %(message)s')
ch = logging.StreamHandler()

ch.setFormatter(fmt)
log.handlers.append(ch)
