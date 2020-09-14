<?php 
session_start();

include 'co1.php';

if($_SERVER['REQUEST_METHOD'] == 'POST'){
   unset($_SESSION['errors']);
   unset($_SESSION['data']);
   
    $pname = filter_var($_POST['projectname'], FILTER_SANITIZE_STRING);
    $keywords = filter_var($_POST['keywords'], FILTER_SANITIZE_STRING);
    $pid= filter_var($_POST['project_id'], FILTER_SANITIZE_STRING);
    $doctor= filter_var($_POST['doctor_name'], FILTER_SANITIZE_STRING);
    $ptrack = filter_var($_POST['track'], FILTER_SANITIZE_STRING);
    $department = filter_var($_POST['department'], FILTER_SANITIZE_STRING);
    $gyear = filter_var($_POST['graduation_year'], FILTER_SANITIZE_STRING);
    $description= filter_var($_POST['description'], FILTER_SANITIZE_STRING); 
    $leadername = filter_var($_POST['leader_name'], FILTER_SANITIZE_STRING);
    $leaderid= filter_var($_POST['leader_id'], FILTER_SANITIZE_STRING);
    $leaderemail= filter_var($_POST['leader_email'], FILTER_SANITIZE_STRING);
    $membersname = filter_var($_POST['members_name'], FILTER_SANITIZE_STRING);

    $error_msg = [];

    if(empty($pname) || strlen($pname) < 10){
        $error_msg[] = 'project name must be more than 10 char';
    }
     if(empty($keywords) ){
        $error_msg[] = 'projec must have key words to descripe it';
    }
    if(empty($pid)) {
        $error_msg[] = 'project id  is required';
       } else if(!is_numeric($pid)) {
        $error_msg[] = 'project id must be numeric';
     } else if(strlen($pid) !=4) {
       $error_msg[] = 'the project id that entered was not 4 digits long';
     }
     else {
    /* Success */
     }
  if($doctor == ''){
        $error_msg[] = 'doctor name is required';
    }
    if($ptrack == ''){
        $error_msg[] = 'track is required';
    }
    if($department == ''){
        $error_msg[] = 'department is required';
    }
    if($gyear == ''){
        $error_msg[] = 'graduation year is required';
    }

    if(empty($description) || strlen($description) < 50){
        $error_msg[] = 'the project description must be more than 50 char';
    }
    if(empty($leadername) || strlen($leadername) < 7){
        $error_msg[] = 'leadername must be more than 7 char';
    }
    if(empty($leaderid)) {
        $error_msg[] = 'leader national id  is required';
       } else if(!is_numeric($leaderid)) {
        $error_msg[] = 'the leader national id must be numeric';
     } else if(strlen($leaderid) !=14) {
       $error_msg[] = 'the leader national id that entered was not 14 digits long';
     }
     else {
    /* Success */
     }
      if(!filter_var($leaderemail, FILTER_VALIDATE_EMAIL)){
        $error_msg[] = 'Enter valid email';
    }
    if(empty($membersname) || strlen($membersname) < 10){
        $error_msg[] = 'members name  must be more than 10 char';
    }



    $fileName = $_FILES["uploadfiles"]['name'];
    $tmpName  = $_FILES["uploadfiles"]['tmp_name'];
    $fileSize = $_FILES["uploadfiles"]['size'];
    $fileType = $_FILES["uploadfiles"]['type'];
    $content=file_get_contents($tmpName);

    $allowed_extension = array('pdf');
 // Get  file extension
    $file_extension = pathinfo($_FILES["uploadfiles"]["name"], PATHINFO_EXTENSION);
    // Validate file input to check if is not empty
    if (! file_exists($_FILES["uploadfiles"]["tmp_name"])) {
       $error_msg[] = 'Choose file to upload';
    }
    // Validate file input to check if is with valid extension
  if (! in_array($file_extension, $allowed_extension)) {
       $error_msg[] ="Upload valid file. Only pdf files are allowed.";   
    } 
     // Validate  file size
  if (($_FILES["uploadfiles"]["size"] > 100000000)) {
         $error_msg[] ="file size exceeds 100MB";
     }
    $data = [
        'pname' => $pname,
        'keywords' => $keywords,
        'pid' => $pid,
        'doctor' => $doctor,
        'ptrack' => $ptrack,
        'department' => $department,
        'gyear' => $gyear,
        'description' => $description,
        'leadername' => $leadername,
        'leaderid' => $leaderid,
        'leaderemail' => $leaderemail,
        'membersname' => $membersname
    ];

    if(empty($error_msg)){
         $sql="SELECT * FROM projects where pid=:pid or pname=:pname";
         $stmt=$conn->prepare($sql);
         $stmt->execute(array('pid'=>$pid,'pname'=>$pname));
         $count=$stmt->rowcount();
        if ($count>0){
            $error_msg[]= "check your project id  that entered was used before";
            $_SESSION['errors'] = $error_msg;
            $_SESSION['data'] = $data;
            header('Location: registration-aaa.php');
            exit();
        }
        else {
           $sql='INSERT INTO projects (`pname`, `keywords`,`pid`, `doctor`, `ptrack`, `department`, `gyear`, `description`)VALUES (?, ?, ?, ?, ?, ?, ?,?)';
           $stmt=$conn->prepare($sql);
           $stmt->execute(array($pname,$keywords, $pid,  $doctor, $ptrack, $department,$gyear, $description ));
           $sql='INSERT INTO project_team( `leadername`, `leaderid`, `leaderemail`, `membersname`,`projectid`)VALUES ( ?,?,?,?,?)';
           $stmt=$conn->prepare($sql);
           $stmt->execute(array( $leadername,$leaderid,$leaderemail,$membersname,$pid));

           $sql='INSERT INTO projectdocument(`pname`, `pr_id`, `document_name`,`pdffile`)VALUES (?,?,?,?)';
           $stmt=$conn->prepare($sql);
           $stmt->execute(array($pname,$pid,$fileName,$content));


     header('Location: successfully.php');
     exit();
    
        }
    }
    else{

        $_SESSION['errors'] = $error_msg;
        $_SESSION['data'] = $data;
        header('Location:registration-aaa.php');
        exit();
    }



}
 ?>