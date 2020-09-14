<?php 
$connect = mysqli_connect("localhost", "root", "", "graduation");
session_start();
$rid = $_POST['r_id'];
//echo $rid;
?>
<!DOCTYPE HTML>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
    <title>Requirments Edit </title>
    

    <link rel="stylesheet" href="css/bootstrap.min.css"/>
    
    <link rel="stylesheet" href="fontawesome/css/all.min.css">
    <link rel="stylesheet" href="css/19.css"/>
    
    
    
    
</head>
<body>
    <!--Start form-->
    <form class="shadow-lg main" method="post" action="req_edit_data.php">
    
        <div class="container-fluid require_head">
            <h4>Your Requirments & Needs</h4>
        </div>

    <!--Strat Add new tracks-->
        <div class="require_body">
            <div class="tracks">
              <h2>Tracks</h2>
            </div>
            
            <hr class="col-md-12">
            <div  > 
             
               <div class="d-none d-lg-block d-xl-block">
             
              <select  class="custom-select" name="track"  id="validationCustom02"  >
                <option value="">Choose Track...</option>
                <option value="Artficial Intelligence">Artficial Intelligence</option>
                <option value="Virtual Reality">Virtual Reality</option>
                <option value="Machine Learning">Machine Learning</option>
                <option value="Image Processing">Image Processing</option>
                <option value="Internet Of Things">Internet Of Things</option>
                <option value="Deep Learning">Deep Learning</option>
                <option value="Web Development">Web Development</option>
                <option value="Android Development">Android Development</option>
                <option value="Desktop Application">Desktop Application</option>
              
              
            </select> 
            <div> 
            
              
            
            </div>
        </div>
    <!--End Add new tracks-->

    <!--Strat Add new tools-->

       
     <!--End Add new tools-->

    <!--Strat Add new description-->
      <?php
      $output = '';  
      $sql = "SELECT * FROM company_requriments where c_id='". $rid."'";  
      $result = mysqli_query($connect, $sql);  
      $row1 = mysqli_fetch_array($result);
      ?>
        <div class="require_body">
            <div class="tracks" id="short_desc">
                <h2>Short description ideas</h2>
            </div>
                
            <hr class="col-md-12">
            <div class="d1">
              <textarea class="form-control" class="input" id="input_description" name ="idea" placeholder="Enter Short description..." rows="3"></textarea>
              <h6 class="short">Please write short, clear description (less than 6 sentences)</h6>
                    
            </div>
                    
         <input type="hidden"  name="view" value=<?php echo $rid; ?> />
        </div>
    <!--End Add new description-->
    <!--Strat footer-->
        
        <div class="container-fluid require_head">
            <button type="button" class="btn btn-outline-secondary back " onclick="document.location ='mycompanyreq.php'">Back</button>
            <button type="submit" class="btn btn-outline-secondary save " >Save</button>
        </div>
    
    <!--End footer-->
   
    
    </form>
    <!--End form-->

    <!--Strat Modal1 -->

      <!--End Modal1 -->

      <!--End Modal3 -->
      <script src="js/jquery-3.4.1.min.js"></script>
      <script src="js/bootstrap.min.js"></script>

</body>
</html>