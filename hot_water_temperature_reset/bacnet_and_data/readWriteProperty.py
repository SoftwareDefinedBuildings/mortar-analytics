import sys, time
import configparser
from threading import Thread, Lock

from bacpypes.debugging import bacpypes_debugging, ModuleLogger
from bacpypes.consolelogging import ConfigArgumentParser

from bacpypes.core import run, stop, deferred
from bacpypes.iocb import IOCB

from bacpypes.pdu import Address
from bacpypes.object import get_object_class, get_datatype

from bacpypes.apdu import SimpleAckPDU, \
    ReadPropertyRequest, ReadPropertyACK, WritePropertyRequest
from bacpypes.primitivedata import Null, Atomic, Boolean, Unsigned, Integer, \
    Real, Double, OctetString, CharacterString, BitString, Date, Time, ObjectIdentifier
from bacpypes.constructeddata import Array, Any, AnyAtomic

from bacpypes.app import BIPSimpleApplication
from bacpypes.local.device import LocalDeviceObject

# some debugging
_debug = 0
_log = ModuleLogger(globals())

# globals
this_application = None

@bacpypes_debugging
class ReadPropertyThread(Thread):

    def request_read(self, args):
        addr, obj_id, prop_id = args[:3]
        obj_id = ObjectIdentifier(obj_id).value
        if prop_id.isdigit():
            prop_id = int(prop_id)

        datatype = get_datatype(obj_id[0], prop_id)
        if not datatype:
            raise ValueError("invalid property for object type")

        # build a request
        request = ReadPropertyRequest(
            objectIdentifier=obj_id,
            propertyIdentifier=prop_id,
            )
        request.pduDestination = Address(addr)

        if len(args) == 4:
            request.propertyArrayIndex = int(args[3])
        if _debug: print("    - request: %r", request)

        return request

    def take_args(self, args):
        self._args = args

    def get_args(self):
        """<addr> <objid> <prop> [ <indx> ]"""
        temp_args = self._args

        addr, obj_type, obj_inst, prop_id = temp_args[:4]
        obj_id = f"{obj_type}:{obj_inst}"

        if len(temp_args) == 5:
            args = (addr, obj_id, prop_id, temp_args[4])
        else:
            args = (addr, obj_id, prop_id)

        return args

    def run(self):
        global valueRead
        valueRead = None

        args = self.get_args()

        try:
            request = self.request_read(args)

            # make an IOCB
            iocb = IOCB(request)
            if _debug: print("    - iocb: %r", iocb)

            # give it to the application
            deferred(this_application.request_io, iocb)

            # wait for it to complete
            iocb.wait()

            # do something for success
            if iocb.ioResponse:
                apdu = iocb.ioResponse

                # should be an ack
                if not isinstance(apdu, ReadPropertyACK):
                    if _debug: print("    - not an ack")
                    return

                # find the datatype
                datatype = get_datatype(apdu.objectIdentifier[0], apdu.propertyIdentifier)
                if _debug: print("    - datatype: %r", datatype)
                if not datatype:
                    raise TypeError("unknown datatype")

                # special case for array parts, others are managed by cast_out
                if issubclass(datatype, Array) and (apdu.propertyArrayIndex is not None):
                    if apdu.propertyArrayIndex == 0:
                        value = apdu.propertyValue.cast_out(Unsigned)
                    else:
                        value = apdu.propertyValue.cast_out(datatype.subtype)
                else:
                    value = apdu.propertyValue.cast_out(datatype)
                if _debug: print("    - value: %r", value)

                #sys.stdout.write(str(value) + '\n')
                valueRead = value
                if hasattr(value, 'debug_contents'):
                    value.debug_contents(file=sys.stdout)
                sys.stdout.flush()

            # do something for error/reject/abort
            if iocb.ioError:
                sys.stdout.write(str(iocb.ioError) + '\n')

        except Exception as error:
            print("exception: %r", error)

        # all done
        stop()


@bacpypes_debugging
class WritePropertyThread(Thread):

    def request_write(self, args):

        addr, obj_id, prop_id = args[:3]
        obj_id = ObjectIdentifier(obj_id).value
        value = args[3]

        indx = None
        if len(args) >= 5:
            if args[4] != "-":
                indx = int(args[4])
        if _debug: print("    - indx: %r", indx)

        priority = None
        if len(args) >= 6:
            priority = int(args[5])
        if _debug: print("    - priority: %r", priority)

        # get the datatype
        datatype = get_datatype(obj_id[0], prop_id)
        if _debug: print("    - datatype: %r", datatype)

        # change atomic values into something encodeable, null is a special case
        if (value == 'null'):
            value = Null()
        elif issubclass(datatype, AnyAtomic):
            dtype, dvalue = value.split(':', 1)
            if _debug: print("    - dtype, dvalue: %r, %r", dtype, dvalue)

            datatype = {
                'b': Boolean,
                'u': lambda x: Unsigned(int(x)),
                'i': lambda x: Integer(int(x)),
                'r': lambda x: Real(float(x)),
                'd': lambda x: Double(float(x)),
                'o': OctetString,
                'c': CharacterString,
                'bs': BitString,
                'date': Date,
                'time': Time,
                'id': ObjectIdentifier,
                }[dtype]
            if _debug: print("    - datatype: %r", datatype)

            value = datatype(dvalue)
            if _debug: print("    - value: %r", value)

        elif issubclass(datatype, Atomic):
            if datatype is Integer:
                value = int(value)
            elif datatype is Real:
                value = float(value)
            elif datatype is Unsigned:
                value = int(value)
            value = datatype(value)
        elif issubclass(datatype, Array) and (indx is not None):
            if indx == 0:
                value = Integer(value)
            elif issubclass(datatype.subtype, Atomic):
                value = datatype.subtype(value)
            elif not isinstance(value, datatype.subtype):
                raise TypeError("invalid result datatype, expecting %s" % (datatype.subtype.__name__,))
        elif not isinstance(value, datatype):
            raise TypeError("invalid result datatype, expecting %s" % (datatype.__name__,))
        if _debug: print("    - encodeable value: %r %s", value, type(value))

        # build a request
        request = WritePropertyRequest(
            objectIdentifier=obj_id,
            propertyIdentifier=prop_id
            )
        request.pduDestination = Address(addr)

        # save the value
        request.propertyValue = Any()
        try:
            request.propertyValue.cast_in(value)
        except Exception as error:
            print("WriteProperty cast error: %r", error)

        # optional array index
        if indx is not None:
            request.propertyArrayIndex = indx

        # optional priority
        if priority is not None:
            request.priority = priority

        if _debug: print("    - request: %r", request)

        return request

    def take_args(self, args):
        self._args = args

    def get_args(self):
        """<addr> <objid> <prop> <value> [ <indx> ] [ <priority> ]"""
        temp_args = self._args

        addr, obj_type, obj_inst, prop_id, value = temp_args[:5]
        obj_id = f"{obj_type}:{obj_inst}"

        if len(temp_args) >= 7:
            args = (addr, obj_id, prop_id, value, temp_args[5], temp_args[6])
        elif len(temp_args) >= 6:
            args = (addr, obj_id, prop_id, value, temp_args[5])
        else:
            args = (addr, obj_id, prop_id, value)

        return args

    def run(self):
        # global valueRead
        valueRead = None

        args = self.get_args()

        try:
            request = self.request_write(args)

            # make an IOCB
            iocb = IOCB(request)
            if _debug: print("    - iocb: %r", iocb)

            # give it to the application
            deferred(this_application.request_io, iocb)

            # wait for it to complete
            iocb.wait()

            # do something for success
            if iocb.ioResponse:
                # should be an ack
                if not isinstance(iocb.ioResponse, SimpleAckPDU):
                    if _debug: print("    - not an ack")
                    return

                sys.stdout.write("ack\n")

            # do something for error/reject/abort
            if iocb.ioError:
                sys.stdout.write(str(iocb.ioError) + '\n')

        except Exception as error:
            print("exception: %r", error)

        # all done
        stop()


#
#   __main__
#

lock_read = Lock()
lock_write = Lock()

def Init(ini_filename):
    global this_application
    bacnet_establish = False

    ### read initialization file
    opts = configparser.ConfigParser()
    opts.read(ini_filename)
    
    objectName = opts.get('BACpypes', 'objectName')
    objectIdentifier = opts.get('BACpypes', 'objectIdentifier')
    maxApduLengthAccepted = opts.get('BACpypes', 'maxApduLengthAccepted')
    segmentationSupported = opts.get('BACpypes', 'segmentationSupported')
    vendorIdentifier = opts.get('BACpypes', 'vendorIdentifier')
    address = opts.get('BACpypes', 'address')

    try:
        if _debug: print("initialization")
        if _debug: print("    - args: %r", (opts))

        # make a device object
        this_device = LocalDeviceObject(
            objectName=objectName,
            objectIdentifier=int(objectIdentifier),
            maxApduLengthAccepted=int(maxApduLengthAccepted),
            segmentationSupported=segmentationSupported,
            vendorIdentifier=int(vendorIdentifier),
            )

        # make a simple application
        this_application = BIPSimpleApplication(this_device, address)

        # get the services supported
        services_supported = this_application.get_services_supported()
        if _debug: print("    - services_supported: %r", services_supported)

        # let the device object know
        # TODO: Currently getting a *** bacpypes.errors.ExecutionError: ('property', 'writeAccessDenied')
        # TODO: Fix the error. Currently does not affect functionality
        # this_device.protocolServicesSupported = services_supported.value
        if _debug: print("after services")

    except Exception as error:
        import pdb; pdb.set_trace()
        if _debug: print("exception: %r", error)
    else:
        bacnet_establish = True

    return bacnet_establish


def read_prop(args):
    with lock_read:
        # create a thread supervisor
        read_property_thread = ReadPropertyThread()

        # input BACnet request arguments
        read_property_thread.take_args(args)

        # start it running when the core is running
        deferred(read_property_thread.start)

        if _debug: print("running")

        run()

        if _debug: print("fini")

    return valueRead


def write_prop(args):
    with lock_write:
        # create a thread supervisor
        write_property_thread = WritePropertyThread()

        # input BACnet request arguments
        write_property_thread.take_args(args)

        # start it running when the core is running
        deferred(write_property_thread.start)

        if _debug: print("running")

        run()

        if _debug: print("fini")

        time.sleep(2)


if __name__ == "__main__":
    BACnet_init_filename = 'BACnet_init_temp_reset.ini'
    access_bacnet = Init(BACnet_init_filename)

    addr = ''
    obj_type = 'analogValue'
    obj_inst = 0
    prop_id = 'presentValue'

    # test read method
    read_args = (addr, obj_type, obj_inst, prop_id)

    print(f"Reading current value...")
    old_value = read_prop(read_args)
    print(old_value)

    # Test write method
    new_value = 70.5

    write_args = (addr, obj_type, obj_inst, prop_id, new_value)
    write_prop(write_args)

    print(f"Wrote new value of {read_prop(read_args)} replacing old value of {old_value}")

    print(f"Switching back to old value...")

    write_prop((addr, obj_type, obj_inst, prop_id, old_value))
    print(f"Confirm that it switched to old value, {read_prop(read_args)}")

    import pdb; pdb.set_trace()
