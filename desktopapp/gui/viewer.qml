import QtQuick
import QtQuick3D
import QtQuick3D.AssetUtils

Rectangle {
    width: 1200
    height: 800
    color: "#202020"
    focus: true

    property real zoomDistance: 150

    View3D {
        anchors.fill: parent

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "#202020"
        }

        // pusat objek
        Node {
            id: target
        }

        // rotasi horizontal
        Node {
            id: yawNode
            parent: target

            // rotasi vertikal
            Node {
                id: pitchNode

                PerspectiveCamera {
                    id: camera

                    position: Qt.vector3d(
                        0,
                        0,
                        zoomDistance
                    )

                    clipNear: 0.1
                    clipFar: 10000
                }
            }
        }

        DirectionalLight {
            eulerRotation.x: -45
            eulerRotation.y: -30
            brightness: 3
        }

        DirectionalLight {
            eulerRotation.x: 45
            eulerRotation.y: 180
            brightness: 2
        }

        RuntimeLoader {
            id: brain

            source: Qt.resolvedUrl(
                "Plastinated_Human_Brain/Plastinated_Human_Brain.gltf"
            )
        }
    }

    MouseArea {
        anchors.fill: parent

        property real lastX
        property real lastY

        acceptedButtons:
            Qt.LeftButton |
            Qt.RightButton

        onPressed:(mouse)=>{

            lastX=mouse.x
            lastY=mouse.y
        }

        onPositionChanged:(mouse)=>{

            let dx=
                mouse.x-lastX

            let dy=
                mouse.y-lastY

            // Orbit 360° penuh
            if(mouse.buttons &
               Qt.LeftButton){

                yawNode.eulerRotation.y
                    += dx*0.5

                pitchNode.eulerRotation.x
                    += dy*0.5
            }

            // Pan
            if(mouse.buttons &
               Qt.RightButton){

                target.position.x
                    -= dx*0.2

                target.position.y
                    += dy*0.2
            }

            lastX=mouse.x
            lastY=mouse.y
        }

        onWheel:(wheel)=>{

            if(
                wheel.angleDelta.y>0
            ){

                zoomDistance*=0.9

            }else{

                zoomDistance*=1.1
            }

            zoomDistance=Math.max(
                5,
                Math.min(
                    zoomDistance,
                    5000
                )
            )

            camera.position=
            Qt.vector3d(
                0,
                0,
                zoomDistance
            )
        }
    }

    Keys.onPressed:(event)=>{

        if(event.key===Qt.Key_R){

            yawNode.eulerRotation=
                Qt.vector3d(
                    0,
                    0,
                    0
                )

            pitchNode.eulerRotation=
                Qt.vector3d(
                    0,
                    0,
                    0
                )

            target.position=
                Qt.vector3d(
                    0,
                    0,
                    0
                )

            zoomDistance=150

            camera.position=
                Qt.vector3d(
                    0,
                    0,
                    zoomDistance
                )
        }
    }

    Rectangle{
        width:260
        height:130
        radius:10
        opacity:0.8

        anchors.left:parent.left
        anchors.top:parent.top
        anchors.margins:10

        color:"#333333"

        Text{
            anchors.centerIn:parent
            color:"white"

            text:
            "Left Drag : Orbit 360°\n"+
            "Right Drag : Pan\n"+
            "Scroll : Zoom\n"+
            "R : Reset"
        }
    }
}