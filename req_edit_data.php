<?php 
session_start();

include 'co1.php';

 

if($_SERVER['REQUEST_METHOD'] == 'POST'){
   unset($_SESSION['errors']);

    $track = filter_var($_POST['track'], FILTER_SANITIZE_STRING);
    $idea= filter_var($_POST['idea'], FILTER_SANITIZE_STRING); 
    $rid= $_POST['view'];
      $error_msg = [];
       if($track == ''||$idea== ''){
        $error_msg[] = 'idea and track are required';
       }
    
        else {
        }
      $da = [
        'track' => $track,
        'idea' => $idea,
        'rid' => $rid
    ];
    if(empty($error_msg)){
      $sql=' update company_requriments set track=:track , idea=:idea where re_id=:rid';
        $stmt=$conn->prepare($sql);
      $stmt->execute($da);
       header('Location:mycompanyreq.php');
     
     exit();
    
    }
    else{

        $_SESSION['errors'] = $error_msg;
        $_SESSION['da'] = $da;
        header('Location:req_edit.php');
        exit();
    }







    }
 ?>
