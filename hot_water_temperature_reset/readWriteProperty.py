
import sys
import configparser
from threading import Thread

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
from bacpypes.constructeddata import Array, Any

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
        temp_args = self._args

        addr, obj_type, obj_inst, prop_id = temp_args[:4]
        obj_id = f"{obj_type}:{obj_inst}"

        if len(temp_args) == 5:
            args = (addr, obj_id, prop_id, temp_args[4])
        else:
            args = (addr, obj_id, prop_id)

        return args

    def run(self):
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

                sys.stdout.write(str(value) + '\n')
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

#
#   __main__
#

def Init(ini_filename):
    global this_application

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
        this_device.protocolServicesSupported = services_supported.value
        if _debug: print("after services")

    except Exception as error:
        if _debug: print("exception: %r", error)

def read_prop(args):
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


if __name__ == "__main__":
    BACnet_init_filename = 'BACnet_init_temp_reset.ini'
    Init(BACnet_init_filename)

    addr = ''
    obj_type = 'analogInput'
    obj_inst = 3000366
    prop_id = 'presentValue'

    args = (addr, obj_type, obj_inst, prop_id)

    read_prop(args)
