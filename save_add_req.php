<?php 
session_start();
 $connect = mysqli_connect("localhost", "root", "", "graduation");
$cid3= $_POST['view'];
include 'co1.php';

if($_SERVER['REQUEST_METHOD'] == 'POST'){
   unset($_SESSION['errors']);
   unset($_SESSION['data']);
   
    $idea = filter_var($_POST['idea'], FILTER_SANITIZE_STRING);
    $track = filter_var($_POST['track'], FILTER_SANITIZE_STRING);
    $error_msg = [];
    $sql2= "SELECT companyname FROM company where company_id ='".$cid3."'";
            $result1 = mysqli_query($connect, $sql2);
            $row1 = mysqli_fetch_row($result1);
           $name= implode(" ",$row1);
        $data = [
        'name' => $name,
        'track' => $track,
        'idea' => $idea,
        'cid3' =>$cid3
        
    ];
    if(empty($error_msg)){   

            
           $sql='INSERT INTO company_requriments(`comanyname`,`track`, `idea`, `c_id`)VALUES (?, ?,? ,?)';
           $stmt=$conn->prepare($sql);
           $stmt->execute(array($name,$track, $idea,$cid3));
           header('Location:ai-cards-explore.php');
     exit();
    
        }

    else{

        $_SESSION['errors'] = $error_msg;
        $_SESSION['data'] = $data;
        header('Location: add_requriment.php');
        exit();
    }



}
 ?>