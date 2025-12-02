# Format: (Code, short, long)

CODES = [
   ("ARR", "Arrival",
     "Arrival of the shipment at a facility, depot, or destination terminal (not necessarily delivered to the consignee). "
     "Typical messages: 'arrival scan', 'arrived at facility', 'arrived at destination depot', 'arrival at sorting center'."),

    ("BCF", "Booking confirmation",
     "The shipment booking with the carrier has been confirmed and accepted by the carrier system. "
     "Typical messages: 'booking confirmed', 'carrier booking accepted', 'shipment booking confirmation received'."),

    ("BCK", "Booking",
     "The shipment has been booked with the carrier or transport has been requested, but may not yet be confirmed. "
     "Typical messages: 'booking created', 'shipment booked', 'pickup booked with carrier', 'transport order created'."),

    ("CAS", "Consignee absence",
     "Delivery attempt failed because the recipient was not available at the delivery address. The shipment was NOT delivered. "
     "Typical messages: 'consignee absent', 'recipient not at home', 'no one available to sign', 'delivery attempted, card left'."),

    ("CND", "Delivery cancelled",
     "The delivery or shipment has been cancelled. No further delivery attempts are planned for this shipment. "
     "Typical messages: 'shipment cancelled', 'delivery cancelled by shipper', 'order cancelled', 'pickup cancelled'."),

    ("CUS", "Customs handling",
     "The shipment is under customs processing, clearance, or inspection (import or export). "
     "Typical messages: 'in customs', 'customs clearance in progress', 'awaiting customs clearance', 'held by customs', 'customs inspection'."),

    ("DAM", "Damage",
     "Damage to the shipment or its packaging has been reported or detected. "
     "Typical messages: 'shipment damaged', 'damaged parcel', 'carton damaged', 'content damaged', 'damage reported'."),

    ("DEL", "Delay",
     "The shipment is delayed compared to the planned schedule (transit, delivery or processing delay). "
     "Typical messages: 'delay in transit', 'delivery delayed', 'operational delay', 'weather delay', 'flight delayed'."),

    ("DEP", "Departed",
     "The shipment has departed from a facility, depot, hub, or terminal (ATD – actual time of departure). "
     "Typical messages: 'departed facility', 'left origin terminal', 'outbound scan', 'departed hub', 'shipment left the depot'."),

    ("ERR", "Error",
     "General error in status processing, data, or transmission. Details can be found in the event information. "
     "Typical messages: 'status error', 'unknown event', 'interface error', 'processing error', 'invalid status received'."),

    ("ETA", "Estimated time of arrival",
     "An expected arrival or delivery date/time has been provided or updated by the carrier. "
     "Typical messages: 'expected delivery tomorrow', 'ETA updated', 'scheduled delivery date', 'estimated arrival time'."),

    ("FER", "Fatal error",
     "A critical or unrecoverable error has occurred in processing or data exchange. Manual intervention is required. "
     "Typical messages: 'fatal error', 'hard error', 'unrecoverable error', 'interface failure', 'message rejected permanently'."),

    ("GIS", "Goods issue",
     "Goods issue has been posted: the shipment has left the shipper’s stock or has been marked as dispatched from the warehouse. "
     "Typical messages: 'goods issue posted', 'shipment left warehouse', 'despatched from warehouse', 'stock reduced'."),

    ("HDO", "Hand over to carrier",
     "The shipment has been physically handed over to the carrier or parcel shop for further transport. "
     "Typical messages: 'handed over to carrier', 'parcel handed to DHL', 'dropped off at parcel shop', 'handover scan at carrier'."),

    ("HIN", "HUB in",
     "The shipment has been received at a carrier hub or sorting center (inbound to hub). "
     "Typical messages: 'arrival at hub', 'inbound scan hub', 'received at sorting center', 'arrived at distribution hub'."),

    ("HOU", "HUB out",
     "The shipment has left a carrier hub or sorting center (outbound from hub) and is on its way to the next facility. "
     "Typical messages: 'departed hub', 'outbound scan hub', 'left sorting center', 'dispatched from distribution hub'."),

    ("HUB", "HUB Handling",
     "Generic handling or processing event at a carrier hub or sorting center (sorting, routing, internal transfer). "
     "Typical messages: 'processed at hub', 'in sorting', 'hub handling', 'at distribution center', 'processed at facility'."),

    ("INF", "Info",
     "General information related to the shipment without a clear change of physical status. All details are in the event text. "
     "Typical messages: 'shipment information updated', 'address information received', 'label created', 'data updated'."),

    ("IOD", "Delivered",
     "Delivered: The shipment was successfully delivered. Key indicators: delivered, zugestellt, abgestellt, POD, signed, left at front door."),

    ("ITR", "In transit",
     "The shipment is in transit between facilities or transport legs (linehaul, air, road, sea). "
     "Typical messages: 'in transit', 'on the way', 'line haul', 'transport by truck', 'shipment en route'."),

    ("MDA", "Missing data",
     "Required shipment data is missing, incomplete, or inconsistent and may prevent normal processing. "
     "Typical messages: 'missing consignee name', 'incomplete address', 'mandatory field missing', 'data incomplete', 'EDI data missing'."),

    ("MIS", "Missing Parts",
     "The whole shipment or parts of it are missing; there is a shortage compared to the expected units or packages. "
     "Typical messages: 'package missing', 'one parcel missing', 'short shipment', 'item not found', 'colli missing'."),

    ("MPW", "Missing paperwork",
     "Required shipment documents are missing, incomplete, or not available for transport or customs. "
     "Typical messages: 'missing invoice', 'no customs documents', 'paperwork missing', 'documents not received'."),

    ("OFD", "Out for Delivery",
     "The shipment has been loaded into the delivery vehicle and is on its way to the recipient, but NOT yet delivered. "
     "Typical messages: 'out for delivery', 'on vehicle for delivery', 'driver has the parcel', 'on the way to recipient'."),

    ("PAK", "Packed",
     "The shipment has been packed or repacked and is ready for handover to the next process step. "
     "Typical messages: 'shipment packed', 'carton packed', 'repacked', 'packing completed'."),

    ("PDL", "Partial Delivery",
     "Only part of the shipment has been delivered; some items or parcels are still missing or in transit. "
     "Typical messages: 'partial delivery', 'one parcel delivered, others pending', 'not all items delivered', 'delivery incomplete'."),

    ("PUP", "Picked Up",
     "The shipment has been collected from the shipper or agreed pickup location by the carrier. "
     "Typical messages: 'picked up from shipper', 'shipment collected', 'pickup completed', 'driver picked up parcel'."),

    ("REF", "Refused by Receiver",
     "The recipient refused to accept the shipment. The shipment was NOT delivered and will be returned or held. "
     "Typical messages: 'delivery refused', 'consignee refused', 'customer rejected shipment', 'receiver declined parcel'."),

    ("RET", "Shipment returned",
     "The shipment is being returned or has been returned to the shipper and will NOT be delivered to the original recipient. "
     "Typical messages: 'returned to sender', 'return to shipper', 'RTS', 'back to origin', 'return shipment'."),

    ("UTD", "Unable to deliver",
     "The carrier was unable to deliver the shipment. The shipment was NOT delivered; another attempt or action is required. "
     "Typical messages: 'unable to deliver', 'delivery attempt failed', 'no access to property', 'delivery not possible', 'door code missing'."),

    ("WAD", "Wrong or incomplete Address",
     "The delivery address is incorrect, incomplete, or cannot be located. The shipment cannot be delivered as addressed. "
     "Typical messages: 'wrong address', 'insufficient address', 'unknown recipient', 'address not found', 'address incomplete'."),

    ("WRN", "Warning",
     "Warning or potential issue related to the shipment or processing that may affect delivery but is not a final failure. "
     "Typical messages: 'warning: address needs verification', 'possible delay', 'hold for clarification', 'risk of delay'."),

]